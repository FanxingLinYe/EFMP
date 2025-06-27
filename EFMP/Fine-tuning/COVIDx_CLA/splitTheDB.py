
import os
import shutil

# 设置路径
base_dir = "/mnt/data0/YXG/NIH_ChestX-ray/"
image_dir = os.path.join(base_dir, "images")
output_dirs = {
    "train": os.path.join(base_dir, "train"),
    "val": os.path.join(base_dir, "val"),
    "test": os.path.join(base_dir, "test")
}
list_files = {
    "train": os.path.join(base_dir, "train_list.txt"),
    "val": os.path.join(base_dir, "val_list.txt"),
    "test": os.path.join(base_dir, "test_list.txt")
}

# 创建输出文件夹（如果不存在）
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# 读取文本文件并复制图像
for split, list_file in list_files.items():
    with open(list_file, 'r') as f:
        for line in f:
            # 提取图像文件名（每行的第一个字段）
            image_name = line.strip().split()[0]
            src_path = os.path.join(image_dir, image_name)
            dst_path = os.path.join(output_dirs[split], image_name)
            
            # 检查源文件是否存在
            if os.path.exists(src_path):
                # 复制文件
                shutil.copy2(src_path, dst_path)
                print(f"Copied {image_name} to {output_dirs[split]}")
            else:
                print(f"Image {image_name} not found in {image_dir}")

