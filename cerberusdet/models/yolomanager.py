import torch
from ultralytics import YOLO
import os
import shutil
from cerberusdet.train import run  # 导入训练逻辑

class YOLOManager:
    def __init__(self, pt_path: str, device: str = "cuda"):
        """
        初始化 YOLOManager 类，管理 .pt 文件路径。
        
        参数:
            pt_path (str): 当前模型的 .pt 文件路径
            device (str): 运行设备，默认为 "cuda"
        """
        self.pt_path = pt_path
        self.device = 0 if torch.cuda.is_available() else "cpu"
    
    def load_model(self, themodel=None) -> YOLO:
        """
        加载 .pt 文件中的模型。
        
        返回:
            YOLO: 加载的 YOLO 模型
        """
        if themodel is None:
            return YOLO(self.pt_path)
        else:
            return YOLO(themodel)
    
    def train(self, 
              data_yaml: str, 
              hyp: str, 
              cfg: str,
              epochs: int = 1, 
              batch_size: int = 16, 
              imgsz: int = 640, 
              project: str = "runs/train", 
              name: str = "exp", 
              exist_ok: bool = False):
        """
        训练模型：加载 .pt 文件，执行训练，训练后更新 .pt 文件。
        
        参数:
            data_yaml (str): 数据集配置文件路径
            hyp (str): 超参数文件路径
            epochs (int): 训练的 epoch 数
            batch_size (int): 批次大小
            imgsz (int): 图像尺寸
            project (str): 保存目录
            name (str): 实验名称
            exist_ok (bool): 是否覆盖现有目录
        """
        # 模拟命令行参数
        opt = {}
        opt["weights"] = self.pt_path
        opt["data"] = data_yaml
        opt["hyp"] = hyp
        opt["cfg"] = cfg
        opt["epochs"] = epochs
        opt["batch-size"] = batch_size
        opt["imgsz"] = imgsz
        opt["resume"] = False
        opt["project"] = project
        opt["name"] = name
        opt["exist-ok"] = exist_ok
        opt["device"] = self.device
        print(self.device)

        new_pt_path = run(opt)
        self.pt_path = new_pt_path  # 更新当前 .pt 文件路径
        print(f"Training completed. Updated model saved to {self.pt_path}")
    
    def get_parameters(self, part: str = "all", themodel=None) -> dict:
        """
        获取模型参数。
        
        参数:
            part (str): 获取哪部分参数，可选 "all"、"backbone"、"neck"、"head"，默认为 "all"
        
        返回:
            dict: 指定部分的 state_dict
        """
        if themodel is None:
            model = self.load_model()
        else:
            model = self.load_model(themodel)
        state_dict = model.model.state_dict()

        # 这里应该做修改
        # TODO:这里应该做修改
        prefixes = {
            "backbone": [
                "blocks.0.model.0.",
                "blocks.0.model.1.",
                "blocks.0.model.2.",
                "blocks.0.model.3.",
                "blocks.0.model.4."
            ],
            "neck": [
                "blocks.0.model.5.",
                "blocks.0.model.6.",
                "blocks.0.model.7.",
                "blocks.0.model.8.",
                "blocks.3.",
                "blocks.6.",
                "blocks.7."
            ],
            "head": [
                "blocks.0.model.9.",
                "blocks.9.",
                "blocks.10.",
                "blocks.12.",
                "blocks.13."
            ]
        }
        

        if part == "all":
            return state_dict
        
        if part not in prefixes:
            raise ValueError(f"Unsupported part: {part}")
        
        part_prefixes = prefixes[part]
        return {k: v for k, v in state_dict.items() if any(k.startswith(p) for p in part_prefixes)}
    
    def update_parameters(self, new_params: dict, part: str = "all"):
        """
        更新模型参数并保存到 .pt 文件。
        
        参数:
            new_params (dict): 新的参数字典（state_dict 格式）
            part (str): 更新哪部分参数，可选 "all"、"backbone"、"neck"、"head"，默认为 "all"
        """
        model = self.load_model()
        current_state = model.model.state_dict()
        
        # 这里应该做修改
        # TODO:这里应该做修改
        prefixes = {
            "backbone": [
                "blocks.0.model.0.",
                "blocks.0.model.1.",
                "blocks.0.model.2.",
                "blocks.0.model.3.",
                "blocks.0.model.4."
            ],
            "neck": [
                "blocks.0.model.5.",
                "blocks.0.model.6.",
                "blocks.0.model.7.",
                "blocks.0.model.8.",
                "blocks.3.",
                "blocks.6.",
                "blocks.7."
            ],
            "head": [
                "blocks.0.model.9.",
                "blocks.9.",
                "blocks.10.",
                "blocks.12.",
                "blocks.13."
            ]
        }
        
        if part == "all":
            current_state.update(new_params)
        else:
            if part not in prefixes:
                raise ValueError(f"Unsupported part: {part}")
            part_prefixes = prefixes[part]
            for k, v in new_params.items():
                if any(k.startswith(p) for p in part_prefixes):
                    current_state[k] = v
        
        model.model.load_state_dict(current_state)
        model.save(self.pt_path)  # 保存更新后的模型
        print(f"Updated {part} parameters and saved to {self.pt_path}")
    
    def send_pt(self, dest_path: str):
        """
        模拟发送 .pt 文件（复制到目标路径）。
        
        参数:
            dest_path (str): 目标路径
        """
        shutil.copy(self.pt_path, dest_path)
        print(f"Sent {self.pt_path} to {dest_path}")
    
    def receive_pt(self, src_path: str):
        """
        模拟接收 .pt 文件（从源路径复制到当前 .pt 路径）。
        
        参数:
            src_path (str): 源路径
        """
        shutil.copy(src_path, self.pt_path)
        print(f"Received {src_path} and updated {self.pt_path}")

# 示例使用
if __name__ == "__main__":
    # 初始化 YOLOManager
    manager = YOLOManager(pt_path="pretrained/yolov8x_state_dict.pt")
    
    # 训练模型
    manager.train(
        data_yaml="data/lightweight_dataset.yaml", 
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml", 
        epochs=1, 
        cfg="cerberusdet/models/yolov8x.yaml")
    
    # manager.train(
    #     data_yaml="data/lightweight_dataset.yaml", 
    #     hyp="data/hyps/hyp.cerber-voc_obj365.yaml", 
    #     epochs=1, 
    #     cfg="cerberusdet/models/yolov8x.yaml")

    # 获取 backbone 参数
    # backbone_params = manager.get_parameters("backbone")
    # print(backbone_params)

    # 模拟接收新的 .pt 文件
    # manager.receive_pt("new_model.pt")
    
    # 更新 backbone 参数（示例）
    # manager.update_parameters(new_backbone_params, "backbone")