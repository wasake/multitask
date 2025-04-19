import os
import zipfile

# 解压 ZIP 文件
# 待解压的 zip 文件
zip_path = "/home/cerberusdet/CerberusDet/data/Object Detection in Images.zip"
# 图片+标签的存储路径
extract_folder = "/home/cerberusdet/CerberusDet/data/extract"
# 标签的存储路径
output_folder = "/home/cerberusdet/CerberusDet/data/output"

def unzip_dataset():
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # 遍历所有 TXT 文件
    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".txt"):
                input_file = os.path.join(root, file)
                relative_path = os.path.relpath(input_file, extract_folder)
                output_file = os.path.join(output_folder, relative_path)

                # 确保输出文件夹结构一致
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                process_txt_file(input_file, output_file)

    print(f"处理完成，所有TXT文件已保存至: {output_folder}")

    # 处理所有 TXT 文件
def process_txt_file(input_path, output_path):
    output_lines = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                x, y, w, h, category = map(str.strip, parts[:5])
                center_x = float(x) + float(w) / 2
                center_y = float(y) - float(h) / 2
                output_lines.append(f"{category} {center_x} {center_y} {w} {h}")

    # 输出到新的 TXT 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))


unzip_dataset()