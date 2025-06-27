
import os
import pandas as pd
from pathlib import Path

def check_missing_files(dataset_path, txt_file_path, output_csv):
    """
    检查 txt 文件中列出的文件在数据集路径中是否存在，并将结果保存为 CSV。
    
    Args:
        dataset_path (str): 数据集根路径 (e.g., /mnt/data0/YXG/COVIDX-7/)
        txt_file_path (str): txt 文件路径 (e.g., train_list_1.txt)
        output_csv (str): 输出 CSV 文件路径
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

    # 存储检查结果
    results = []
    missing_count = 0

    # 遍历 txt 文件中的每一行
    for line in lines:
        # 分割路径和标签
        parts = line.strip().split()
        if len(parts) != 2:
            print(f"无效行格式: {line.strip()}")
            continue
        
        file_path, label = parts
        # 构建完整文件路径
        full_path = dataset_path / file_path
        
        # 检查文件是否存在
        exists = full_path.exists()
        if not exists:
            missing_count += 1
        
        # 记录结果
        results.append({
            'file_path': file_path,
            'label': label,
            'exists': 'Yes' if exists else 'No'
        })

    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 保存到 CSV
    df.to_csv(output_csv, index=False)
    print(f"检查完成！缺失文件数量: {missing_count}")
    print(f"结果已保存到: {output_csv}")

def main():
    # 数据集根路径
    dataset_path = "/mnt/data0/YXG/COVIDX-7/"
    
    # 要检查的 txt 文件列表
    txt_files = [
        "/mnt/data0/YXG/COVIDX-7/train_list_1.txt",
        "/mnt/data0/YXG/COVIDX-7/train_list_10.txt",
        "/mnt/data0/YXG/COVIDX-7/train_list.txt",
        "/mnt/data0/YXG/COVIDX-7/test_list.txt",
        "/mnt/data0/YXG/COVIDX-7/val_list.txt",
    ]
    
    # 输出文件夹
    output_dir = "check_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历并检查每个 txt 文件
    for txt_file in txt_files:
        output_csv = os.path.join(output_dir, f"result_{os.path.basename(txt_file)}.csv")
        print(f"\n正在检查 {txt_file}...")
        check_missing_files(dataset_path, txt_file, output_csv)

if __name__ == "__main__":
    main()

