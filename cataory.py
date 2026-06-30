import json
from collections import Counter

json_path = "/media/keshuo/新加卷/yyc/archive/dataset/train_data_ocr.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 80)
print(f"总数据量: {len(data)}")

# 提取 category_final 字段
categories = [item["category"] for item in data]

category_count = Counter(categories)

print(f"类别数量: {len(category_count)}")
print("\n每类数量及类别名称：")
for cat, count in sorted(category_count.items()):
    print(f"类别名称: {cat}  →  数量: {count}")

# 单独列出所有类别名字
print("\n所有类别名称:", list(category_count.keys()))