
import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert_dcm_to_png(input_folder, output_folder):
    """
    将指定文件夹中的 .dcm 文件转换为 .png 格式，并保存到输出文件夹。
    
    Args:
        input_folder (str): 包含 .dcm 文件的输入文件夹路径
        output_folder (str): 保存 .png 文件的输出文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件夹中的所有 .dcm 文件
    dcm_files = [f for f in os.listdir(input_folder) if f.endswith('.dcm')]
    
    # 使用 tqdm 显示进度条
    for dcm_file in tqdm(dcm_files, desc="Converting .dcm to .png"):
        try:
            # 读取 DICOM 文件
            dcm_path = os.path.join(input_folder, dcm_file)
            dcm = pydicom.dcmread(dcm_path)
            
            # 获取像素数据并归一化到 0-255
            pixel_array = dcm.pixel_array
            pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
            pixel_array = pixel_array.astype(np.uint8)
            
            # 转换为 PIL 图像
            image = Image.fromarray(pixel_array)
            
            # 如果是单通道图像，转换为 RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 保存为 .png 文件
            output_path = os.path.join(output_folder, dcm_file.replace('.dcm', '.png'))
            image.save(output_path)
            
        except Exception as e:
            print(f"Error processing {dcm_file}: {str(e)}")
            continue

def main():
    base_path = "/mnt/data0/YXG/RSNA Pneumonia Detection/"
    train_input_folder = os.path.join(base_path, "stage_2_train_images")
    test_input_folder = os.path.join(base_path, "stage_2_test_images")
    train_output_folder = os.path.join(base_path, "stage_2_train_images_png")
    test_output_folder = os.path.join(base_path, "stage_2_test_images_png")
    
    # 转换训练集
    print("Converting training images...")
    convert_dcm_to_png(train_input_folder, train_output_folder)
    
    # 转换测试集
    print("Converting test images...")
    convert_dcm_to_png(test_input_folder, test_output_folder)

if __name__ == "__main__":
    main()

