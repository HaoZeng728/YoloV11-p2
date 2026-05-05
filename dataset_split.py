# dataset_split.py
import os
import random
import shutil
from pathlib import Path

# ==================== 请修改这里的路径 ====================
IMAGES_DIR = "Visdrone\VisDrone2019-DET-val\VisDrone2019-DET-val\images"  # 图像文件夹
LABELS_DIR = "Visdrone\VisDrone2019-DET-val\VisDrone2019-DET-val\labels"  # 标签文件夹
OUTPUT_DIR = "datasets"  # 输出文件夹

TRAIN_RATIO = 0.8   # 训练集比例
VAL_RATIO = 0.15    # 验证集比例
TEST_RATIO = 0.05   # 测试集比例
# =========================================================

def main():
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (Path(OUTPUT_DIR) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (Path(OUTPUT_DIR) / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(IMAGES_DIR).glob(ext))
    image_files = sorted(image_files)
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 筛选出有对应标签的图像
    valid_images = []
    for img in image_files:
        label_path = Path(LABELS_DIR) / f"{img.stem}.txt"
        if label_path.exists():
            valid_images.append(img)
    
    print(f"有效图像（有标签）: {len(valid_images)} 张")
    
    # 打乱顺序
    random.seed(42)
    random.shuffle(valid_images)
    
    # 计算数量
    total = len(valid_images)
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    
    # 划分
    train_images = valid_images[:train_count]
    val_images = valid_images[train_count:train_count+val_count]
    test_images = valid_images[train_count+val_count:]
    
    print(f"\n训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")
    print(f"测试集: {len(test_images)} 张")
    
    # 复制文件
    print("\n正在复制文件...")
    
    for img in train_images:
        label = Path(LABELS_DIR) / f"{img.stem}.txt"
        shutil.copy2(img, Path(OUTPUT_DIR) / 'images' / 'train' / img.name)
        shutil.copy2(label, Path(OUTPUT_DIR) / 'labels' / 'train' / label.name)
    
    for img in val_images:
        label = Path(LABELS_DIR) / f"{img.stem}.txt"
        shutil.copy2(img, Path(OUTPUT_DIR) / 'images' / 'val' / img.name)
        shutil.copy2(label, Path(OUTPUT_DIR) / 'labels' / 'val' / label.name)
    
    for img in test_images:
        label = Path(LABELS_DIR) / f"{img.stem}.txt"
        shutil.copy2(img, Path(OUTPUT_DIR) / 'images' / 'test' / img.name)
        shutil.copy2(label, Path(OUTPUT_DIR) / 'labels' / 'test' / label.name)
    
    print("\n完成！")
    print(f"输出目录: {Path(OUTPUT_DIR).absolute()}")

if __name__ == "__main__":
    main()