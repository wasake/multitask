import os
import shutil
from pathlib import Path

def is_file_valid(file_path, label_path, logic_type='range', n=1):
    """
    判断逻辑函数：检查文件名并决定文件放入train还是val。
    参数:
        file_path (str): 图片文件路径 (jpg)
        label_path (str): 对应标签文件路径 (txt)
        logic_type (str): 逻辑类型，'range'（1-333到train，其余到val）或'multiple'（n的倍数到train，其余到val）
        n (int): 当logic_type='multiple'时，检查是否为n的倍数
    返回:
        str: 'train'（放入train文件夹）或'val'（放入val文件夹），若文件无效返回None
    """
    # 检查文件是否存在
    # if not os.path.exists(file_path) or not os.path.exists(label_path):
    #     return None
    
    # 检查标签文件是否为空
    # with open(label_path, 'r') as f:
    #     content = f.read().strip()
    #     if not content:  # 如果标签文件为空，跳过
    #         return None
    
    # 提取文件名编号（假设格式为0000001.jpg到0000999.jpg）
    file_name = os.path.basename(file_path)  # 获取文件名，如0000001.jpg
    try:
        file_number = int(file_name.split('.')[0])  # 提取编号，如1
    except ValueError:
        print(f"警告：文件名 {file_name} 格式不正确，跳过")
        return None
    

    # 此处为划分train/val判断逻辑
    # 逻辑1：编号1-333放入train，其余放入val
    if logic_type == 'range':
        return 'val' if 1 <= file_number <= 333 else 'train'
    
    # 逻辑2：编号为n的倍数放入train，其余放入val
    elif logic_type == 'multiple':
        return 'val' if file_number % n == 0 else 'train'
    
    else:
        raise ValueError("logic_type必须是'range'或'multiple'")
    
    return None

def split_dataset_train_val(input_dir, output_dir, logic_type='range', n=1, image_ext='.jpg', label_ext='.txt'):
    """
    将数据集按判断逻辑划分为train和val，复制到输出文件夹的train/和val/子文件夹。
    参数:
        input_dir (str): 输入文件夹路径，包含images/和labels/子文件夹
        output_dir (str): 输出文件夹路径，将创建train/和val/子文件夹
        logic_type (str): 逻辑类型，'range'（1-333到train，其余到val）或'multiple'（n的倍数到train，其余到val）
        n (int): 当logic_type='multiple'时，检查是否为n的倍数
        image_ext (str): 图片文件扩展名，默认为.jpg
        label_ext (str): 标签文件扩展名，默认为.txt
    """
    # 转换为Path对象
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 输入文件夹路径
    input_images_dir = input_dir / 'images'
    input_labels_dir = input_dir / 'labels'
    
    # 输出文件夹路径（train和val）
    train_images_dir = output_dir / 'train' / 'images'
    train_labels_dir = output_dir / 'train' / 'labels'
    val_images_dir = output_dir / 'val' / 'images'
    val_labels_dir = output_dir / 'val' / 'labels'
    
    # 创建输出文件夹
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件夹
    if not input_images_dir.exists() or not input_labels_dir.exists():
        raise FileNotFoundError("输入文件夹必须包含images/和labels/子文件夹")
    
    # 获取所有图片文件
    image_files = [f for f in input_images_dir.iterdir() if f.suffix.lower() == image_ext]
    
    train_count = 0
    val_count = 0
    skipped_count = 0
    
    for img_path in image_files:
        # 对应的标签文件路径
        label_path = input_labels_dir / img_path.with_suffix(label_ext).name
        
        # 调用判断逻辑
        split_type = is_file_valid(img_path, label_path, logic_type, n)
        
        if split_type == 'train':
            # 复制到train文件夹
            shutil.copy(img_path, train_images_dir / img_path.name)
            
            # 存在无label情况
            if os.path.exists(label_path):
                shutil.copy(label_path, train_labels_dir / label_path.name)
            train_count += 1
        elif split_type == 'val':
            # 复制到val文件夹
            shutil.copy(img_path, val_images_dir / img_path.name)
            
            # 存在无label情况
            if os.path.exists(label_path):
                shutil.copy(label_path, val_labels_dir / label_path.name)
            val_count += 1
        else:
            skipped_count += 1
    
    print(f"处理完成：训练集 {train_count} 对文件，验证集 {val_count} 对文件，跳过 {skipped_count} 对文件")
    print(f"训练集输出路径：{train_images_dir.parent}")
    print(f"验证集输出路径：{val_images_dir.parent}")

if __name__ == "__main__":
    # 示例用法
    input_folder = "/home/cerberusdet/CerberusDet/data/UAV2"  # 输入文件夹，包含images/和labels/
    output_folder = "/home/cerberusdet/CerberusDet/data/UAV2"  # 输出文件夹，将创建images/和labels/
    
    # 示例1：1-333到train，其余到val
    # print("逻辑1：编号1-333到train，其余到val：")
    # split_dataset_train_val(input_folder, output_folder + "_range", logic_type='range')
    
    # 示例2：编号为5的倍数到train，其余到val
    # print("\n逻辑2：编号为5的倍数到train，其余到val：")
    split_dataset_train_val(input_folder, output_folder, logic_type='multiple', n=5)