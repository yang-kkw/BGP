import os
import json
import re
import csv
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =====================================================
# 1. 路径配置：按顺序先跑 Kaggle，再跑 BookCover30
# =====================================================
ROOT = "/media/keshuo/新加卷/yyc/archive"
CLIP_PATH = os.path.join(ROOT, "clip-large")

DATASETS = [
    {
        "name": "Kaggle",
        "json_path": os.path.join(ROOT, "dataset/clean_data_ocr.json"),
        "img_root": os.path.join(ROOT, "dataset/clean_images"),
        "weight_path": os.path.join(ROOT, "best_clip_large_final_kaggle.pth"),
        "csv_path": os.path.join(ROOT, "per_class_recall_f1_kaggle.csv"),
        "category_key": "category_final",
        "expected_acc": 70.44,
    },
    {
        "name": "BookCover30",
        "json_path": os.path.join(ROOT, "dataset/train_data_ocr.json"),
        "img_root": os.path.join(ROOT, "dataset/train_images"),
        "weight_path": os.path.join(ROOT, "best_clip_large_final_bookcover30.pth"),
        "csv_path": os.path.join(ROOT, "per_class_recall_f1_bookcover30.csv"),
        "category_key": "category",
        "expected_acc": 70.80,
    },
]


# =====================================================
# 2. 评估参数：必须与训练时的数据划分一致
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 4
SEED = 42
TEST_SIZE = 0.1
LORA_R = 8
LORA_ALPHA = 16


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =====================================================
# 3. OCR 清洗：与训练代码保持一致
# =====================================================
def clean_ocr(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    words = [word for word in text.split() if len(word) > 1]
    return " ".join(words[:30])


# =====================================================
# 4. Tokenizer
# =====================================================
tokenizer = CLIPTokenizer.from_pretrained(
    CLIP_PATH,
    local_files_only=True,
)


# =====================================================
# 5. 测试集 Dataset：不使用任何随机增强
# =====================================================
class CLIPDataset(Dataset):
    def __init__(self, data: List[dict], root: str):
        self.data = data
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.48145466, 0.4578275, 0.40821073],
                [0.26862954, 0.26130258, 0.27577711],
            ),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        image_name = item["image_name"]
        image_path = (
            image_name
            if os.path.isabs(image_name)
            else os.path.join(self.root, image_name)
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        title = item.get("title", "")
        ocr_clean = clean_ocr(item.get("ocr_text", ""))
        text = f"a photo of a book cover titled {title}. text: {ocr_clean}"

        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        # 训练代码直接使用 item["label"]，因此评估时也保持完全一致。
        label = int(item["label"])

        return image, input_ids, attention_mask, label


# =====================================================
# 6. LoRA 与 SE Block：必须与训练时完全一致
# =====================================================
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16):
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
    def __init__(self, dim: int = 1536, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# =====================================================
# 7. 模型结构：与生成两个权重的训练代码保持一致
# =====================================================
class CLIPLoRA(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            CLIP_PATH,
            local_files_only=True,
        )

        for parameter in self.clip.parameters():
            parameter.requires_grad = False

        for name, parameter in self.clip.vision_model.named_parameters():
            if "layers.22" in name or "layers.23" in name:
                parameter.requires_grad = True

        for name, parameter in self.clip.text_model.named_parameters():
            if "layers.10" in name or "layers.11" in name:
                parameter.requires_grad = True

        self.clip.visual_projection = LoRALinear(
            self.clip.visual_projection,
            r=LORA_R,
            alpha=LORA_ALPHA,
        )
        self.clip.text_projection = LoRALinear(
            self.clip.text_projection,
            r=LORA_R,
            alpha=LORA_ALPHA,
        )

        self.se = SEBlock(dim=1536, reduction=16)

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

        features = torch.cat([image_features, text_features], dim=1)
        features = self.se(features)
        return self.classifier(features)


# =====================================================
# 8. 类别名称映射
# =====================================================
def get_category_name(item: dict, category_key: str, label: int) -> str:
    candidate_keys = [
        category_key,
        "category_final",
        "category_name",
        "genre",
        "genre_name",
        "label_name",
    ]

    for key in candidate_keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()

    return str(label)


def build_id_to_category(
    data: List[dict],
    category_key: str,
    num_classes: int,
) -> Dict[int, str]:
    id_to_category: Dict[int, str] = {}

    for item in data:
        label = int(item["label"])
        if label not in id_to_category:
            id_to_category[label] = get_category_name(
                item=item,
                category_key=category_key,
                label=label,
            )

    for label in range(num_classes):
        id_to_category.setdefault(label, str(label))

    return id_to_category


# =====================================================
# 9. 收集测试集预测
# =====================================================
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for images, input_ids, attention_mask, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attention_mask = attention_mask.to(DEVICE, non_blocking=True)

            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(images, input_ids, attention_mask)

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    return np.asarray(all_labels), np.asarray(all_predictions)


# =====================================================
# 10. 单个数据集评估
# =====================================================
def evaluate_one_dataset(config: dict) -> dict:
    dataset_name = config["name"]

    print("\n" + "=" * 110)
    print(f"Evaluate dataset: {dataset_name}")
    print("=" * 110)

    for required_path_key in ("json_path", "img_root", "weight_path"):
        path = config[required_path_key]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{dataset_name}: 路径不存在 -> {required_path_key}: {path}"
            )

    with open(config["json_path"], "r", encoding="utf-8") as file:
        data = json.load(file)

    labels_all = [int(item["label"]) for item in data]
    unique_labels = sorted(set(labels_all))
    num_classes = len(unique_labels)

    # 训练代码直接使用原始整数标签，因此必须是从0开始的连续整数。
    expected_labels = list(range(num_classes))
    if unique_labels != expected_labels:
        raise ValueError(
            f"{dataset_name}: 标签不是从0开始的连续整数。\n"
            f"实际标签: {unique_labels}\n"
            f"期望标签: {expected_labels}\n"
            "请不要自行重排标签，否则会与已保存权重的分类器输出节点错位。"
        )

    _, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels_all,
    )

    print(f"Total samples : {len(data)}")
    print(f"Num classes   : {num_classes}")
    print(f"Test samples  : {len(test_data)}")
    print(f"Weight path   : {config['weight_path']}")

    test_loader = DataLoader(
        CLIPDataset(test_data, config["img_root"]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    model = CLIPLoRA(num_classes=num_classes).to(DEVICE)

    state_dict = torch.load(
        config["weight_path"],
        map_location=DEVICE,
    )

  
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    y_true, y_pred = collect_predictions(model, test_loader)

    overall_acc = accuracy_score(y_true, y_pred)
    overall_macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    id_to_category = build_id_to_category(
        data=data,
        category_key=config["category_key"],
        num_classes=num_classes,
    )

    print("\nOverall Results")
    print(f"Acc       : {overall_acc * 100:.2f}%")
    print(f"Macro-F1  : {overall_macro_f1 * 100:.2f}%")

    expected_acc = config.get("expected_acc")
    if expected_acc is not None:
        difference = abs(overall_acc * 100 - expected_acc)
        if difference <= 0.20:
            print(
                f"Check     : Acc与论文结果 {expected_acc:.2f}% 基本一致 "
                f"(差值 {difference:.2f} 个百分点)"
            )
        else:
            print(
                f"WARNING   : 当前Acc与论文结果 {expected_acc:.2f}% 相差 "
                f"{difference:.2f} 个百分点，请检查权重、JSON内容、标签和测试集划分。"
            )

    print("\nPer-class Recall and F1")
    print("-" * 100)
    print(
        f"{'ID':>4}  {'Genre':<45} "
        f"{'Support':>9}  {'Recall/%':>10}  {'F1/%':>10}"
    )
    print("-" * 100)

    rows = []
    for class_id in range(num_classes):
        genre_name = id_to_category[class_id]
        row = {
            "Class_ID": class_id,
            "Genre": genre_name,
            "Support": int(support[class_id]),
            "Recall/%": round(float(recall[class_id] * 100), 2),
            "F1/%": round(float(class_f1[class_id] * 100), 2),
        }
        rows.append(row)

        print(
            f"{class_id:>4}  {genre_name:<45} "
            f"{int(support[class_id]):>9}  "
            f"{recall[class_id] * 100:>10.2f}  "
            f"{class_f1[class_id] * 100:>10.2f}"
        )

    print("-" * 100)

    with open(
        config["csv_path"],
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Class_ID", "Genre", "Support", "Recall/%", "F1/%"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved  : {config['csv_path']}")

  
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "acc": overall_acc,
        "macro_f1": overall_macro_f1,
        "csv_path": config["csv_path"],
    }


# =====================================================
# 11. 主程序：两个数据集依次评估
# =====================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    summaries = []

    for dataset_config in DATASETS:
        result = evaluate_one_dataset(dataset_config)
        summaries.append(result)

    print("\n" + "=" * 110)
    print("Final Summary")
    print("=" * 110)
    print(f"{'Dataset':<20} {'Acc/%':>12} {'Macro-F1/%':>15}  CSV")

    for result in summaries:
        print(
            f"{result['dataset']:<20} "
            f"{result['acc'] * 100:>12.2f} "
            f"{result['macro_f1'] * 100:>15.2f}  "
            f"{result['csv_path']}"
        )
