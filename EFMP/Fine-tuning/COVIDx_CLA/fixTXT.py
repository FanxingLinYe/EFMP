
import os
from pathlib import Path

def filter_missing_files(dataset_path, txt_file_path, output_txt):
    """
    检查 txt 文件中列出的文件是否存在，过滤掉缺失文件，生成新的 txt 文件。
    
    Args:
        dataset_path (str): 数据集根路径 (e.g., /mnt/data0/YXG/COVIDX-7/)
        txt_file_path (str): 输入 txt 文件路径 (e.g., train_list_1.txt)
        output_txt (str): 输出 txt 文件路径
    """
    # 确保数据集路径存在
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"数据集路径 {dataset_path} 不存在！")
        return

    # 读取 txt 文件
    try:
        with open(txt_file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"txt 文件 {txt_file_path} 不存在！")
        return

    # 统计原始数据量
    total_files = len(lines)
    valid_lines = []
    missing_count = 0

    # 遍历 txt 文件中的每一行
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"无效行格式: {line.strip()}")
            continue
        
        file_path, label = parts
        # 构建完整文件路径
        full_path = dataset_path / file_path
        
        # 检查文件是否存在
        if full_path.exists():
            valid_lines.append(line)
        else:
            missing_count += 1
            print(f"发现缺失文件: {file_path}")

    # 统计过滤后的数据量
    remaining_files = len(valid_lines)

    # 保存新的 txt 文件
    with open(output_txt, 'w') as f:
        f.writelines(valid_lines)

    # 打印统计信息
    print(f"\n检查完成：{txt_file_path}")
    print(f"原始数据量: {total_files}")
    print(f"缺失文件数量: {missing_count}")
    print(f"过滤后数据量: {remaining_files}")
    print(f"新的 txt 文件已保存到: {output_txt}")

def main():
    # 数据集根路径
    dataset_path = "/mnt/data0/YXG/COVIDX-7/"
    
    # 要处理的 txt 文件列表
    txt_files = [
        "/mnt/data0/YXG/COVIDX-7/train_list_1.txt",
        "/mnt/data0/YXG/COVIDX-7/train_list_10.txt",
        "/mnt/data0/YXG/COVIDX-7/train_list.txt",
        "/mnt/data0/YXG/COVIDX-7/test_list.txt",
        "/mnt/data0/YXG/COVIDX-7/val_list.txt",
    ]
    
    # 输出文件夹
    output_dir = "filtered_txt"
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历并处理每个 txt 文件
    for txt_file in txt_files:
        output_txt = os.path.join(output_dir, f"filtered_{os.path.basename(txt_file)}")
        print(f"\n正在处理 {txt_file}...")
        filter_missing_files(dataset_path, txt_file, output_txt)

if __name__ == "__main__":
    main()

