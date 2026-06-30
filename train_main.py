import os
import json
import re
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import CLIPModel, CLIPTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =====================================================
# 1. 路径配置：两个数据集
# =====================================================
DATASETS = [
    {
        "name": "Kaggle",
        "json_path": "/media/keshuo/新加卷/yyc/archive/dataset/clean_data_ocr.json",
        "img_root": "/media/keshuo/新加卷/yyc/archive/dataset/clean_images"
    },
    {
        "name": "BookCover30",
        "json_path": "/media/keshuo/新加卷/yyc/archive/dataset/train_data_ocr.json",
        "img_root": "/media/keshuo/新加卷/yyc/archive/dataset/train_images"
    }
]

clip_path = "/media/keshuo/新加卷/yyc/archive/clip-large"


# =====================================================
# 2. 训练参数
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8
epochs = 25
num_workers = 4

label_smoothing = 0.15
gradient_accumulation_steps = 4

weight_decay = 3e-2
patience = 5
seed = 42

lora_r = 8
lora_alpha = 16


# =====================================================
# 3. 固定随机种子
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(seed)


# =====================================================
# 4. OCR 清洗
# =====================================================
def clean_ocr(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [w for w in text.split() if len(w) > 1]

    return " ".join(words[:30])


# =====================================================
# 5. CLIP Tokenizer
# =====================================================
tokenizer = CLIPTokenizer.from_pretrained(clip_path)


# =====================================================
# 6. Dataset
# =====================================================
class CLIPDataset(Dataset):
    def __init__(self, data, root, label2id, train=True):
        self.data = data
        self.root = root
        self.label2id = label2id
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
               
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_name = item["image_name"]
        img_path = img_name if os.path.isabs(img_name) else os.path.join(self.root, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        title = item.get("title", "")
        ocr = item.get("ocr_text", "")
        ocr_clean = clean_ocr(ocr)

        text = f"a photo of a book cover titled {title}. text: {ocr_clean}"

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]

        label = self.label2id[item["label"]]

        return image, input_ids, attention_mask, label


# =====================================================
# 7. LoRA
# =====================================================
class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()

        self.linear = linear
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.scale = alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale


# =====================================================
# 8. SE Block
# =====================================================
class SEBlock(nn.Module):
    def __init__(self, dim=1536, reduction=16):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


# =====================================================
# 9. CLIP-Large + LoRA + SE 模型
# =====================================================
class CLIPLoRASE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_path)

        # 先冻结 CLIP 全部参数
        for p in self.clip.parameters():
            p.requires_grad = False

        # 视觉编码器只解冻最后两层
        for name, param in self.clip.vision_model.named_parameters():
            if "layers.22" in name or "layers.23" in name:
                param.requires_grad = True

        # 文本编码器只解冻最后两层
        for name, param in self.clip.text_model.named_parameters():
            if "layers.10" in name or "layers.11" in name:
                param.requires_grad = True

        print("✅ CLIP-Large 图像编码器最后2层、文本编码器最后2层已解冻")

        # 投影层加入 LoRA
        self.clip.visual_projection = LoRALinear(
            self.clip.visual_projection,
            r=lora_r,
            alpha=lora_alpha
        )

        self.clip.text_projection = LoRALinear(
            self.clip.text_projection,
            r=lora_r,
            alpha=lora_alpha
        )

        # SE 融合模块
        self.se = SEBlock(dim=1536, reduction=16)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.65),
            nn.Linear(768, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.clip.get_image_features(images)

        txt_feat = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        feat = torch.cat([img_feat, txt_feat], dim=1)
        feat = self.se(feat)

        logits = self.classifier(feat)

        return logits


# =====================================================
# 10. 只取可训练参数
# =====================================================
def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


# =====================================================
# 11. 评估函数：Acc + Macro-F1
# =====================================================
def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for img, ids, mask, label in loader:
            img = img.to(device)
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            pred = model(img, ids, mask).argmax(dim=1)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return acc, f1


# =====================================================
# 12. 单个数据集训练
# =====================================================
def run_one_dataset(dataset_cfg):
    dataset_name = dataset_cfg["name"]
    json_path = dataset_cfg["json_path"]
    img_root = dataset_cfg["img_root"]

    print("\n" + "=" * 90)
    print(f"Start Experiment | Model: CLIP-Large + LoRA + SE | Dataset: {dataset_name}")
    print("=" * 90)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_all = [d["label"] for d in data]
    unique_labels = sorted(list(set(labels_all)))

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {len(data)}")
    print(f"Num classes: {num_classes}")

    train_data, val_data = train_test_split(
        data,
        test_size=0.1,
        random_state=seed,
        stratify=labels_all
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(val_data)}")

    train_loader = DataLoader(
        CLIPDataset(train_data, img_root, label2id, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        CLIPDataset(val_data, img_root, label2id, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = CLIPLoRASE(num_classes).to(device)
    scaler = GradScaler(enabled=(device == "cuda"))

    optimizer = torch.optim.AdamW([
        {"params": trainable_params(model.clip.vision_model), "lr": 5e-6},
        {"params": trainable_params(model.clip.text_model), "lr": 5e-6},
        {"params": trainable_params(model.clip.visual_projection), "lr": 2e-5},
        {"params": trainable_params(model.clip.text_projection), "lr": 2e-5},
        {"params": model.se.parameters(), "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 8e-5},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_acc = 0.0
    best_f1 = 0.0
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        optimizer.zero_grad()

        for step, (img, ids, mask, label) in enumerate(train_loader):
            img = img.to(device)
            ids = ids.to(device)
            mask = mask.to(device)
            label = label.to(device)

            with autocast(enabled=(device == "cuda")):
                out = model(img, ids, mask)
                loss = criterion(out, label)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps

        # 处理最后不足 gradient_accumulation_steps 的 batch
        if (step + 1) % gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        acc, f1 = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss {avg_loss:.4f} | "
            f"Acc {acc * 100:.2f}% | "
            f"F1 {f1 * 100:.2f}%"
        )

        scheduler.step(acc)

        # 按 Acc 保存最优，F1 记录该最佳 Acc 对应的 F1
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            wait = 0

            save_path = f"best_clip_large_lora_se_{dataset_name}.pth"
            torch.save(model.state_dict(), save_path)

            print(
                f"✅ BEST | "
                f"Acc {best_acc * 100:.2f}% | "
                f"F1 {best_f1 * 100:.2f}% | "
                f"Saved: {save_path}"
            )
        else:
            wait += 1
            if wait >= patience:
                print("⛔ Early stop")
                break

    print("\n" + "-" * 90)
    print(f"Final Result | Dataset: {dataset_name}")
    print(f"Best Acc: {best_acc * 100:.2f}%")
    print(f"Best F1: {best_f1 * 100:.2f}%")
    print("-" * 90)

    return {
        "dataset": dataset_name,
        "acc": best_acc,
        "f1": best_f1
    }


# =====================================================
# 13. 主程序：两个数据集依次跑
# =====================================================
if __name__ == "__main__":
    print("Device:", device)

    all_results = []

    for dataset_cfg in DATASETS:
        result = run_one_dataset(dataset_cfg)
        all_results.append(result)

    print("\n" + "=" * 90)
    print("Overall Performance Summary")
    print("=" * 90)

    print(f"{'Model':<28} {'Dataset':<18} {'Acc/%':<12} {'F1/%':<12}")

    for r in all_results:
        print(
            f"{'CLIP-Large+LoRA+SE':<28} "
            f"{r['dataset']:<18} "
            f"{r['acc'] * 100:<12.2f} "
            f"{r['f1'] * 100:<12.2f}"
        )