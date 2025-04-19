import os
from PIL import Image  # 使用 Pillow 读取图片

# 目录路径
txt_folder ="D:\BaiduNetdiskDownload\Object Detection in Images原始\VisDrone2019-DET-train\VisDrone2019-DET-train\\annotations" # 假设txt文件存放在VDtrain/annotations
image_folder = "D:\BaiduNetdiskDownload\Object Detection in Images原始\VisDrone2019-DET-train\VisDrone2019-DET-train\images"  # 假设图像存放在VDtrain/images

# 遍历txt文件
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(txt_folder, txt_file)
        image_path = os.path.join(image_folder, txt_file.replace(".txt", ".jpg"))  # 修改扩展名匹配图片格式

        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"找不到对应的图片: {image_path}")
            continue
        
        # 读取图片尺寸（不使用 OpenCV）
        with Image.open(image_path) as img:
            img_width, img_height = img.size  # 获取图像宽度和高度
        
        # 读取并解析txt文件
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            # data = list(map(int, line.strip().split(",")))  # 使用逗号分割并将每个元素转换为整数
            data = list(map(int, filter(None, line.strip().split(" "))))  # 过滤掉空字符串

            # 提取边界框数据
            bbox_left = data[0]
            bbox_top = data[1]
            bbox_width = data[2]
            bbox_height = data[3]

            category = data[5]
            
            # 计算中心点和尺寸占比
            center_x = (bbox_left + bbox_width / 2) / img_width
            center_y = (bbox_top + bbox_height / 2) / img_height
            width_ratio = bbox_width / img_width
            height_ratio = bbox_height / img_height
            
            # 生成新格式
            new_line = f"{category} {center_x:.6f} {center_y:.6f} {width_ratio:.6f} {height_ratio:.6f}\n"
            new_lines.append(new_line)

        # 写回txt文件
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

print("处理完成！")
