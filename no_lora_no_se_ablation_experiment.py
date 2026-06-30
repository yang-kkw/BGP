import os
import json
import re
import gc
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
# 5. label 排序，避免 0,1,10,11,2 这种顺序
# =====================================================
def sort_label_key(x):
    try:
        return int(x)
    except Exception:
        return str(x)


# =====================================================
# 6. CLIP Tokenizer
# =====================================================
tokenizer = CLIPTokenizer.from_pretrained(
    clip_path,
    local_files_only=True
)


# =====================================================
# 7. Dataset
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

        # w/o SE + w/o LoRA：仍然保留标题与 OCR 文本
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
# 8. CLIP-Large，同时去掉 LoRA 和 SE
# =====================================================
class CLIPNoLoRANoSE(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            clip_path,
            local_files_only=True
        )

        # 先冻结 CLIP 的全部参数。
        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        # 与完整模型保持一致：视觉编码器仅解冻最后两层。
        for name, parameter in self.clip.vision_model.named_parameters():
            if "layers.22" in name or "layers.23" in name:
                parameter.requires_grad = True

        # 与完整模型保持一致：文本编码器仅解冻最后两层。
        for name, parameter in self.clip.text_model.named_parameters():
            if "layers.10" in name or "layers.11" in name:
                parameter.requires_grad = True

        print("✅ 已解冻 CLIP-Large 图像编码器最后2层和文本编码器最后2层")
        print("✅ w/o LoRA：视觉投影层和文本投影层保持原始冻结状态")
        print("✅ w/o SE：图像与文本特征直接拼接后送入分类器")

        # 不使用 LoRA：不替换 visual_projection 和 text_projection。
        # 不使用 SE：不定义 self.se。
        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.65),
            nn.Linear(768, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.clip.get_image_features(pixel_values=images)

        txt_feat = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        # w/o SE：归一化后的图像、文本特征直接拼接。
        feat = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(feat)


# =====================================================
# 9. 消融配置检查
# =====================================================
def check_ablation(model):
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )

    lora_names = [name for name in trainable_names if "lora_" in name.lower()]
    se_names = [
        name for name in trainable_names
        if name.startswith("se.") or ".se." in name
    ]

    assert not hasattr(model, "se"), "错误：w/o SE 模型中仍定义了 SE 模块"
    assert len(lora_names) == 0, "错误：w/o LoRA 模型中仍存在 LoRA 参数"
    assert len(se_names) == 0, "错误：w/o SE 模型中仍存在 SE 参数"

    # 原始 CLIP 投影层应保持冻结，防止无 LoRA 实验意外训练投影层。
    for parameter in model.clip.visual_projection.parameters():
        assert not parameter.requires_grad, "错误：visual_projection 在 w/o LoRA 实验中被解冻"
    for parameter in model.clip.text_projection.parameters():
        assert not parameter.requires_grad, "错误：text_projection 在 w/o LoRA 实验中被解冻"

    print(f"Trainable parameters: {trainable_count:,}")
    print("Ablation check passed: no LoRA, no SE")


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

    f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return acc, f1


# =====================================================
# 12. 单个数据集训练
# =====================================================
def run_one_dataset(dataset_cfg):
    dataset_name = dataset_cfg["name"]
    json_path = dataset_cfg["json_path"]
    img_root = dataset_cfg["img_root"]

    print("\n" + "=" * 90)
    print(f"Start Ablation | Model: CLIP-Large + w/o LoRA + w/o SE | Dataset: {dataset_name}")
    print("=" * 90)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels_all = [d["label"] for d in data]
    unique_labels = sorted(list(set(labels_all)), key=sort_label_key)

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

    model = CLIPNoLoRANoSE(num_classes).to(device)
    check_ablation(model)
    scaler = GradScaler(enabled=(device == "cuda"))

    optimizer = torch.optim.AdamW([
        # 与完整模型一致，继续微调视觉/文本编码器最后两层。
        {"params": trainable_params(model.clip.vision_model), "lr": 5e-6},
        {"params": trainable_params(model.clip.text_model), "lr": 5e-6},

        # w/o LoRA：不加入视觉和文本投影层参数。
        # w/o SE：不加入 SE 模块参数。
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

    save_path = f"best_clip_large_no_lora_no_se_{dataset_name}.pth"

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

        # 为了和最终模型保持一致，仍然按 Acc 保存最优
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            wait = 0

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
    print(f"Final Ablation Result | Dataset: {dataset_name} | w/o LoRA + w/o SE")
    print(f"Best Acc: {best_acc * 100:.2f}%")
    print(f"Best F1: {best_f1 * 100:.2f}%")
    print("-" * 90)

    del model
    del train_loader
    del val_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    print("Ablation Summary | w/o LoRA + w/o SE")
    print("=" * 90)

    print(f"{'Model':<32} {'Dataset':<18} {'Acc/%':<12} {'F1/%':<12}")

    for r in all_results:
        print(
            f"{'CLIP-Large w/o LoRA+SE':<32} "
            f"{r['dataset']:<18} "
            f"{r['acc'] * 100:<12.2f} "
            f"{r['f1'] * 100:<12.2f}"
        )