import os

#目标路径
txt_folder = "/home/cerberusdet/CerberusDet/data/VD/labels/VDval"
out_folder = "/home/cerberusdet/CerberusDet/data/VD/labels_after/VDval"

os.makedirs(out_folder, exist_ok=True)
for txt_file in os.listdir(txt_folder):
    if txt_file.endswith(".txt"):
        txt_path=os.path.join(txt_folder,txt_file)
        out_path=os.path.join(out_folder,txt_file)
        with open(txt_path,"r",encoding="utf-8") as f:
            lines=f.readlines()
        new_lines=[]
        for line in lines:
            data=list(map(float,filter(None,line.strip().split(" "))))

            if data[0] in [0,3,5,6,7,8,9,10,11,12,13,14,15]:
                continue
            if data[0]==4:
                data[0]=0
            if data[0]==2:
                data[0]=1
            data[0]=int(data[0])
            new_line=f"{data[0]} {data[1]:.6f} {data[2]:.6f} {data[3]:.6f} {data[4]:.6f} \n"
            new_lines.append(new_line)

        with open(out_path,"w",encoding="utf-8") as f:
            f.writelines(new_lines)
print("处理完成！")
