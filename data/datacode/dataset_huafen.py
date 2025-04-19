import os
import shutil
import random
from pathlib import Path

def is_file_valid(file_path, label_path):
    """
    判断逻辑函数：用户可在此定义是否复制文件的条件。
    参数:
        file_path (str): 图片文件路径 (jpg)
        label_path (str): 对应标签文件路径 (txt)
    返回:
        bool: True表示复制该文件，False表示跳过
    """
    # 示例判断逻辑：检查文件是否存在且标签文件不为空
    # if not os.path.exists(file_path) or not os.path.exists(label_path):
    #     return False
    
    # 检查标签文件是否有内容（非空）
    # with open(label_path, 'r') as f:
    #     content = f.read().strip()
    #     if not content:  # 如果标签文件为空，跳过
    #         print("test empty")
    #         return False
    
    # 示例：可以添加更多条件，例如：
    # - 检查图片尺寸
    # - 检查标签文件中的类别ID是否在预期范围内
    # - 随机选择部分文件（例如80%复制）
    # if random.random() > 0.8:  # 随机跳过20%的文件
    #     return False

    # 提取文件名编号（假设格式为0000001.jpg到0000999.jpg）
    file_name = os.path.basename(file_path)  # 获取文件名，如0000001.jpg
    try:
        file_number = int(file_name.split('.')[0])  # 提取编号，如1
    except ValueError:
        print(f"警告：文件名 {file_name} 格式不正确，跳过")
        return False
    
    if 667 <= file_number <= 999:
        return True
    else:
        return False

def split_dataset(input_dir, output_dir, image_ext='.jpg', label_ext='.txt'):
    """
    将数据集从输入文件夹复制到输出文件夹，期间调用判断逻辑。
    参数:
        input_dir (str): 输入文件夹路径，包含images/和labels/子文件夹
        output_dir (str): 输出文件夹路径，将创建images/和labels/子文件夹
        image_ext (str): 图片文件扩展名，默认为.jpg
        label_ext (str): 标签文件扩展名，默认为.txt
    """
    # 转换为Path对象，方便处理路径
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 输入文件夹中的images和labels路径
    input_images_dir = input_dir / 'images'
    input_labels_dir = input_dir / 'labels'
    
    # 输出文件夹中的images和labels路径
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    
    # 创建输出文件夹
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件夹是否存在
    if not input_images_dir.exists() or not input_labels_dir.exists():
        raise FileNotFoundError("输入文件夹必须包含images/和labels/子文件夹")
    
    # 获取所有图片文件
    image_files = [f for f in input_images_dir.iterdir() if f.suffix.lower() == image_ext]
    
    copied_count = 0
    skipped_count = 0
    
    for img_path in image_files:
        # 对应的标签文件路径

        label_path = input_labels_dir / img_path.with_suffix(label_ext).name
        
        # 调用判断逻辑
        if is_file_valid(img_path, label_path):
            # 复制图片和标签文件到输出文件夹
            shutil.copy(img_path, output_images_dir / img_path.name)

            # 存在无label情况
            if os.path.exists(label_path):
                shutil.copy(label_path, output_labels_dir / label_path.name)
            copied_count += 1
        else:
            skipped_count += 1
    
    print(f"处理完成：复制 {copied_count} 对文件，跳过 {skipped_count} 对文件")
    print(f"输出路径：{output_dir}")

if __name__ == "__main__":
    # 示例用法
    input_folder = "/home/cerberusdet/CerberusDet/data/federal_UAV"  # 输入文件夹，包含images/和labels/
    output_folder = "/home/cerberusdet/CerberusDet/data/UAV2"  # 输出文件夹，将创建images/和labels/
    
    # 调用划分函数
    try:
        split_dataset(input_folder, output_folder)
    except Exception as e:
        print(f"错误：{e}")