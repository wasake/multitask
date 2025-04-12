import torch
from ultralytics import YOLO
import os
import shutil
from cerberusdet.models.yolomanager import YOLOManager  # 假设 yolomanager.py 在此路径

class NetworkEndpoint():
    def __init__(self, pt_path: str, data_yaml: str, hyp: str, cfg: str, device: str = "cuda"):
        """
        初始化通讯节点。
        
        参数:
            pt_path (str): 初始模型的 .pt 文件路径
            data_yaml (str): 数据集配置文件路径
            hyp (str): 超参数文件路径
            device (str): 运行设备，默认为 "cuda"
        """
        self.pt_path = pt_path
        self.data_yaml = data_yaml
        self.hyp = hyp
        self.cfg = cfg
        self.device = device if torch.cuda.is_available() else "cpu"
        self.yolo_manager = YOLOManager(pt_path, device)

    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, project: str = "runs/train", name: str = "exp", exist_ok: bool = False):
        """
        训练模型。
        
        参数:
            epochs (int): 训练的 epoch 数
            batch_size (int): 批次大小
            imgsz (int): 图像尺寸
            project (str): 保存目录
            name (str): 实验名称
            exist_ok (bool): 是否覆盖现有目录
        """
        self.yolo_manager.train(
            data_yaml=self.data_yaml,
            hyp=self.hyp,
            cfg=self.cfg,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            project=project,
            name=name,
            exist_ok=exist_ok
        )
        self.pt_path = self.yolo_manager.pt_path  # 更新 .pt 文件路径

    def send_pt(self, dest_path: str):
        """
        发送 .pt 文件。
        
        参数:
            dest_path (str): 目标路径
        """
        self.yolo_manager.send_pt(dest_path)

    def receive_pt(self, src_path: str):
        """
        接收 .pt 文件。
        
        参数:
            src_path (str): 源路径
        """
        self.yolo_manager.receive_pt(src_path)
        self.pt_path = self.yolo_manager.pt_path  # 更新 .pt 文件路径

class Client(NetworkEndpoint):
    def __init__(self, server, pt_path: str, data_yaml: str, hyp: str, cfg: str, device: str = "cuda"):
        """
        初始化客户端。
        
        参数:
            server (Server): 关联的服务器对象
            pt_path (str): 初始模型的 .pt 文件路径
            data_yaml (str): 数据集配置文件路径
            hyp (str): 超参数文件路径
            device (str): 运行设备，默认为 "cuda"
        """
        super().__init__(pt_path, data_yaml, hyp, cfg, device)
        self.server = server
    
    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, project: str = "clientruns/train", name: str = "client", exist_ok: bool = False):
        super().train(epochs, batch_size, imgsz, project, name, exist_ok)

    def transmit(self, total_epochs: int = 10, transmit_epochs: int = 1):
        """
        训练并与服务器通信。
        
        参数:
            transmit_epochs (int): 每n次 epoch 后，进行一次通信
            total_epochs (int): 总共进行多少epoch，应为transmit_epochs的整数倍
        """
        if total_epochs % transmit_epochs != 0:
            total_epochs = transmit_epochs * (total_epochs // transmit_epochs + 1)

        for _ in range(0, total_epochs // transmit_epochs):
            # Step 1: 训练本地模型
            self.train(epochs=transmit_epochs)
            self.server.train(epochs=transmit_epochs)

            # Step 2: 发送 .pt 文件给服务器
            server_received_path = os.path.join(os.path.dirname(self.server.pt_path), "client_model.pt")
            self.send_pt(server_received_path)
            

            # Step 3: 接收服务器发回的聚合后的 .pt 文件
            aggregated_pt_path = self.server.transmit(self)
            self.receive_pt(aggregated_pt_path)

class Server(NetworkEndpoint):
    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, project: str = "serverruns/train", name: str = "server", exist_ok: bool = False):
        super().train(epochs, batch_size, imgsz, project, name, exist_ok)

    def aggregate(self, client_pt_path: str):
        """
        聚合客户端和服务器的 backbone 参数。
        
        参数:
            client_pt_path (str): 客户端的 .pt 文件路径
        
        返回:
            str: 聚合后的 .pt 文件路径
        """
        # 加载客户端模型
        client_backbone_params = self.yolo_manager.get_parameters(part="backbone", themodel=client_pt_path)  # 使用 YOLOManager 获取 backbone 参数
        
        # 加载服务器模型
        server_model = self.yolo_manager.load_model()
        server_backbone_params = self.yolo_manager.get_parameters(part="backbone")
        
        # 对 backbone 参数取平均
        averaged_backbone = {
            k: (client_backbone_params[k] + server_backbone_params[k]) / 2
            for k in client_backbone_params.keys()
        }
        
        # 更新服务器模型的 backbone 参数
        self.yolo_manager.update_parameters(averaged_backbone, part="backbone")
        
        # 保存聚合后的模型
        aggregated_pt_path = os.path.join(os.path.dirname(self.pt_path), "aggregated_model.pt")
        server_model.save(aggregated_pt_path)
        return aggregated_pt_path

    def transmit(self, client):
        """
        与客户端通信：接收 .pt 文件，聚合参数并发回。
        
        参数:
            client (Client): 客户端对象
        
        返回:
            str: 聚合后的 .pt 文件路径
        """
        # Step 1: 从客户端接收 .pt 文件
        client_pt_path = os.path.join(os.path.dirname(self.pt_path), "client_model.pt")
        
        # Step 2: 聚合参数
        aggregated_pt_path = self.aggregate(client_pt_path)
        
        # Step 3: 将聚合后的 .pt 文件发回客户端
        client_received_path = os.path.join(os.path.dirname(client.pt_path), "aggregated_model.pt")
        shutil.copy(aggregated_pt_path, client_received_path)
        return client_received_path
    
if __name__ == "__main__":
    # 初始化服务器
    server = Server(
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/lightweight_dataset.yaml",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )
    
    # 初始化客户端
    client = Client(
        server=server,
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/lightweight_dataset.yaml",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )
    
    # 客户端训练并与服务器通信
    client.transmit(total_epochs=1)