
import argparse
import csv
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =====================================================
# 1. 路径配置
# =====================================================
DATASETS: List[Dict[str, str]] = [
    {
        "name": "Kaggle",
        "json_path": "/media/keshuo/新加卷/yyc/archive/dataset/clean_data_ocr.json",
        "img_root": "/media/keshuo/新加卷/yyc/archive/dataset/clean_images",
    },
    {
        "name": "BookCover30",
        "json_path": "/media/keshuo/新加卷/yyc/archive/dataset/train_data_ocr.json",
        "img_root": "/media/keshuo/新加卷/yyc/archive/dataset/train_images",
    },
]

CLIP_PATH = "/media/keshuo/新加卷/yyc/archive/clip-large"


# =====================================================
# 2. 训练参数（与完整模型保持一致）
# =====================================================
BATCH_SIZE = 8
EPOCHS = 25
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.15
GRAD_ACC_STEPS = 4
WEIGHT_DECAY = 3e-2
EARLY_STOP_PATIENCE = 5
SEED = 42
LORA_R = 8
LORA_ALPHA = 16

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLIP-Large 消融实验")
    parser.add_argument(
        "--ablation",
        choices=["full", "no_lora", "no_se", "no_ocr", "all_ablations"],
        default="no_ocr",
        help="选择实验类型；all_ablations 会依次运行 no_lora、no_se、no_ocr。",
    )
    parser.add_argument(
        "--dataset",
        choices=["Kaggle", "BookCover30", "all"],
        default="all",
        help="选择数据集。",
    )
    parser.add_argument(
        "--output_dir",
        default="ablation_outputs",
        help="模型权重和结果文件保存目录。",
    )
    return parser.parse_args()


def set_seed(seed: int = SEED) -> None:
    """固定随机性，使不同消融设置尽可能可比。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_ocr(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [word for word in text.split() if len(word) > 1]
    return " ".join(words[:30])


TOKENIZER = CLIPTokenizer.from_pretrained(CLIP_PATH)


class CLIPDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        root: str,
        label2id: Dict[str, int],
        train: bool = True,
        use_ocr: bool = True,
    ) -> None:
        self.data = data
        self.root = root
        self.label2id = label2id
        self.use_ocr = use_ocr

        common = [
            transforms.Resize((224, 224)),
        ]
        if train:
            common.append(transforms.ColorJitter(0.1, 0.1, 0.1))
        common.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.transform = transforms.Compose(common)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        img_name = item["image_name"]
        img_path = img_name if os.path.isabs(img_name) else os.path.join(self.root, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        title = item.get("title", "")
        if self.use_ocr:
            ocr_clean = clean_ocr(item.get("ocr_text", ""))
            text = f"a photo of a book cover titled {title}. text: {ocr_clean}"
        else:
            # 去掉 OCR 后仍保留标题，确保只移除 OCR 信息而不是整个文本模态。
            text = f"a photo of a book cover titled {title}."

        encoded = TOKENIZER(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        label = self.label2id[item["label"]]
        return (
            image,
            encoded["input_ids"][0],
            encoded["attention_mask"][0],
            label,
        )


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.linear = linear
        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)
        self.scale = alpha / r

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale


class SEBlock(nn.Module):
    def __init__(self, dim: int = 1536, reduction: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class CLIPAblationModel(nn.Module):
    def __init__(self, num_classes: int, use_lora: bool, use_se: bool) -> None:
        super().__init__()
        self.use_lora = use_lora
        self.use_se = use_se
        self.clip = CLIPModel.from_pretrained(CLIP_PATH)

        # 先冻结全部 CLIP 参数。
        for param in self.clip.parameters():
            param.requires_grad = False

        # 与完整模型一致：视觉和文本编码器仅解冻最后两层。
        for name, param in self.clip.vision_model.named_parameters():
            if "layers.22" in name or "layers.23" in name:
                param.requires_grad = True

        for name, param in self.clip.text_model.named_parameters():
            if "layers.10" in name or "layers.11" in name:
                param.requires_grad = True

        if use_lora:
            self.clip.visual_projection = LoRALinear(
                self.clip.visual_projection, r=LORA_R, alpha=LORA_ALPHA
            )
            self.clip.text_projection = LoRALinear(
                self.clip.text_projection, r=LORA_R, alpha=LORA_ALPHA
            )

        self.fusion = SEBlock(dim=1536, reduction=16) if use_se else nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(1536, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.65),
            nn.Linear(768, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.clip.get_image_features(pixel_values=images)
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        fused = torch.cat([image_features, text_features], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


def trainable_parameters(module: nn.Module) -> List[nn.Parameter]:
    return [param for param in module.parameters() if param.requires_grad]


def add_group(groups: List[Dict], params: Iterable[nn.Parameter], lr: float) -> None:
    params = list(params)
    if params:
        groups.append({"params": params, "lr": lr})


def build_optimizer(model: CLIPAblationModel) -> torch.optim.Optimizer:
    groups: List[Dict] = []
    add_group(groups, trainable_parameters(model.clip.vision_model), 5e-6)
    add_group(groups, trainable_parameters(model.clip.text_model), 5e-6)

    # 仅当启用 LoRA 时，投影层中才存在可训练的 A/B 矩阵。
    add_group(groups, trainable_parameters(model.clip.visual_projection), 2e-5)
    add_group(groups, trainable_parameters(model.clip.text_projection), 2e-5)

    # no_se 时 fusion=Identity，没有参数，自动跳过。
    add_group(groups, trainable_parameters(model.fusion), 5e-5)
    add_group(groups, model.classifier.parameters(), 8e-5)

    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    predictions: List[int] = []
    labels: List[int] = []

    with torch.no_grad():
        for images, input_ids, attention_mask, target in loader:
            images = images.to(DEVICE, non_blocking=True)
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)

            logits = model(images, input_ids, attention_mask)
            pred = logits.argmax(dim=1)

            predictions.extend(pred.cpu().tolist())
            labels.extend(target.tolist())

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    return acc, macro_f1


def experiment_flags(ablation: str) -> Dict[str, bool]:
    return {
        "use_lora": ablation != "no_lora",
        "use_se": ablation != "no_se",
        "use_ocr": ablation != "no_ocr",
    }


def run_one_dataset(dataset_cfg: Dict[str, str], ablation: str, output_dir: str) -> Dict:
    # 每个实验重新固定随机种子，保证划分、初始化和样本顺序一致。
    set_seed(SEED)
    flags = experiment_flags(ablation)

    dataset_name = dataset_cfg["name"]
    print("\n" + "=" * 96)
    print(
        f"Experiment: {ablation} | Dataset: {dataset_name} | "
        f"LoRA={flags['use_lora']} | SE={flags['use_se']} | OCR={flags['use_ocr']}"
    )
    print("=" * 96)

    with open(dataset_cfg["json_path"], "r", encoding="utf-8") as file:
        data = json.load(file)

    labels_all = [item["label"] for item in data]
    unique_labels = sorted(set(labels_all))
    label2id = {label: index for index, label in enumerate(unique_labels)}

    train_data, test_data = train_test_split(
        data,
        test_size=0.1,
        random_state=SEED,
        stratify=labels_all,
    )

    print(f"Total={len(data)} | Train={len(train_data)} | Test={len(test_data)}")
    print(f"Classes={len(unique_labels)} | Device={DEVICE}")

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        CLIPDataset(
            train_data,
            dataset_cfg["img_root"],
            label2id,
            train=True,
            use_ocr=flags["use_ocr"],
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=generator,
    )
    test_loader = DataLoader(
        CLIPDataset(
            test_data,
            dataset_cfg["img_root"],
            label2id,
            train=False,
            use_ocr=flags["use_ocr"],
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = CLIPAblationModel(
        num_classes=len(unique_labels),
        use_lora=flags["use_lora"],
        use_se=flags["use_se"],
    ).to(DEVICE)

    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_acc = -1.0
    best_f1 = -1.0
    best_epoch = 0
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for step, (images, input_ids, attention_mask, target) in enumerate(train_loader):
            images = images.to(DEVICE, non_blocking=True)
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(images, input_ids, attention_mask)
                loss = criterion(logits, target) / GRAD_ACC_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACC_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * GRAD_ACC_STEPS

        if (step + 1) % GRAD_ACC_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = total_loss / len(train_loader)
        acc, macro_f1 = evaluate(model, test_loader)
        scheduler.step(acc)

        print(
            f"Epoch {epoch + 1:02d} | Loss {avg_loss:.4f} | "
            f"Acc {acc * 100:.2f}% | Macro-F1 {macro_f1 * 100:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            best_f1 = macro_f1
            best_epoch = epoch + 1
            wait = 0

            weight_path = os.path.join(
                output_dir,
                f"best_{dataset_name}_{ablation}.pth",
            )
            #torch.save(model.state_dict(), weight_path)
            print(f"BEST -> {weight_path}")
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "ablation": ablation,
        "use_lora": flags["use_lora"],
        "use_se": flags["use_se"],
        "use_ocr": flags["use_ocr"],
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "num_classes": len(unique_labels),
        "best_epoch": best_epoch,
        "acc": best_acc,
        "f1_macro": best_f1,
        "acc_percent": round(best_acc * 100, 2),
        "f1_macro_percent": round(best_f1 * 100, 2),
        "seed": SEED,
    }

    print("-" * 96)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def save_results(results: List[Dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"ablation_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "ablation_results.csv")
    fieldnames = [
        "timestamp",
        "dataset",
        "ablation",
        "use_lora",
        "use_se",
        "use_ocr",
        "train_samples",
        "test_samples",
        "num_classes",
        "best_epoch",
        "acc",
        "f1_macro",
        "acc_percent",
        "f1_macro_percent",
        "seed",
    ]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nJSON results: {json_path}")
    print(f"CSV summary : {csv_path}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation == "all_ablations":
        ablations = ["no_lora", "no_se", "no_ocr"]
    else:
        ablations = [args.ablation]

    if args.dataset == "all":
        datasets = DATASETS
    else:
        datasets = [item for item in DATASETS if item["name"] == args.dataset]

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results: List[Dict] = []
    for ablation in ablations:
        for dataset_cfg in datasets:
            results.append(run_one_dataset(dataset_cfg, ablation, args.output_dir))

    save_results(results, args.output_dir)

    print("\n" + "=" * 96)
    print(f"{'Dataset':<16} {'Ablation':<12} {'Acc/%':>10} {'Macro-F1/%':>14}")
    print("-" * 96)
    for result in results:
        print(
            f"{result['dataset']:<16} {result['ablation']:<12} "
            f"{result['acc_percent']:>10.2f} {result['f1_macro_percent']:>14.2f}"
        )
    print("=" * 96)


if __name__ == "__main__":
    main()
