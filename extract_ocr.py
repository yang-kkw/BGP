import os
import json
import easyocr
from tqdm import tqdm

# ======================
# 路径配置
# ======================
json_path = "/media/keshuo/新加卷/yyc/archive/dataset/train_data.json"
img_root = "/media/keshuo/新加卷/yyc/archive/dataset/train_images"
save_path = "/media/keshuo/新加卷/yyc/archive/dataset/train_data_ocr.json"

# ======================
# 初始化 OCR
# ======================
reader = easyocr.Reader(['en'], gpu=True)

# ======================
# 读取原始数据
# ======================
print("📖 读取 JSON 文件...")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"原始数据量: {len(data)} 条")

# ======================
# 处理数据：清理缺失图片 + 修改字段名 + OCR
# ======================
new_data = []
missing_count = 0
processed_count = 0

for item in tqdm(data, desc="处理进度"):
    # 提取文件名（去掉 images/ 前缀）
    img_filename = os.path.basename(item["image_path"])
    img_path = os.path.join(img_root, img_filename)
    
    # 检查图片是否存在
    if not os.path.exists(img_path):
        missing_count += 1
        # 缺失的图片跳过，不加入新数据
        continue
    
    # 创建新条目：保留所有原始字段
    new_item = {}
    
    # 复制所有原始字段
    for key, value in item.items():
        if key == "image_path":
            # 将 image_path 改为 image_name，只保留文件名
            new_item["image_name"] = img_filename
        else:
            # 保留其他所有字段（label, title, author, category 等）
            new_item[key] = value
    
    # 添加 OCR 文本字段
    try:
        result = reader.readtext(img_path, detail=0)
        new_item["ocr_text"] = " ".join(result)
        processed_count += 1
    except Exception as e:
        print(f"\n⚠️ OCR 失败: {img_filename} - {e}")
        new_item["ocr_text"] = ""
    
    new_data.append(new_item)

# ======================
# 保存结果
# ======================
print("\n💾 保存结果...")
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

# ======================
# 输出统计信息
# ======================
print("\n" + "="*60)
print("✅ 处理完成！")
print("="*60)
print(f"原始数据: {len(data)} 条")
print(f"缺失图片: {missing_count} 条（已跳过）")
print(f"成功处理: {processed_count} 条")
print(f"最终保存: {len(new_data)} 条")
print(f"保存路径: {save_path}")

# 显示第一条数据示例
if new_data:
    print("\n📋 数据格式示例：")
    print(json.dumps(new_data[0], ensure_ascii=False, indent=2))