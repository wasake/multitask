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


class FederationManager:
    def __init__(self, server, clients):
        """
        初始化联邦学习管理器。

        参数:
            server (Server): 服务器对象
            clients (list): 客户端对象列表
        """
        self.server = server
        self.clients = clients
        self.client_pt_paths = []

    def collect_client_pts(self):
        """收集所有客户端的模型路径"""
        self.client_pt_paths = [client.pt_path for client in self.clients]

    def aggregate_and_distribute(self, server_trains=True):
        """
        触发服务器聚合并将结果分发给客户端。

        参数:
            server_trains (bool): 服务器是否训练自己的模型并参与聚合
        """
        if server_trains:
            self.server.train()
            server_pt_path = self.server.pt_path
        else:
            server_pt_path = None

        # 聚合所有模型
        aggregated_pt_path = self.server.aggregate(self.client_pt_paths, server_pt_path)

        # 分发聚合后的模型给所有客户端
        for client in self.clients:
            client.receive_pt(aggregated_pt_path)

class Server(NetworkEndpoint):
    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, 
              project: str = "/root/autodl-tmp/serverruns/train", name: str = "server", 
              exist_ok: bool = False):
        super().train(epochs, batch_size, imgsz, project, name, exist_ok)

    def aggregate(self, client_pt_paths, server_pt_path=None):
        """
        聚合多个客户端和服务器（可选）的 backbone 参数。

        参数:
            client_pt_paths (list): 客户端的 .pt 文件路径列表
            server_pt_path (str, optional): 服务器的 .pt 文件路径

        返回:
            str: 聚合后的 .pt 文件路径
        """
        # 加载所有客户端的 backbone 参数
        client_backbones = [
            self.yolo_manager.get_parameters(part="backbone", themodel=pt_path) 
            for pt_path in client_pt_paths
        ]

        # 如果服务器参与训练，加载其 backbone 参数
        if server_pt_path:
            server_backbone = self.yolo_manager.get_parameters(part="backbone", themodel=server_pt_path)
            all_backbones = client_backbones + [server_backbone]
        else:
            all_backbones = client_backbones

        # 对所有 backbone 参数取平均
        averaged_backbone = {}
        for key in all_backbones[0].keys():
            averaged_backbone[key] = sum([backbone[key] for backbone in all_backbones]) / len(all_backbones)

        # 更新服务器模型的 backbone 参数
        self.yolo_manager.update_parameters(averaged_backbone, part="backbone")

        # 保存聚合后的模型
        aggregated_pt_path = os.path.join(os.path.dirname(self.pt_path), "aggregated_model.pt")
        self.yolo_manager.save_model(aggregated_pt_path)
        return aggregated_pt_path

class Client(NetworkEndpoint):
    def __init__(self, pt_path: str, data_yaml: str, hyp: str, cfg: str, device: str = "cuda"):
        """
        初始化客户端（移除对服务器的直接依赖）。

        参数:
            pt_path (str): 初始模型的 .pt 文件路径
            data_yaml (str): 数据集配置文件路径
            hyp (str): 超参数文件路径
            cfg (str): 模型配置文件路径
            device (str): 运行设备，默认为 "cuda"
        """
        super().__init__(pt_path, data_yaml, hyp, cfg, device)

    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, 
              project: str = "/root/autodl-fs/clientruns/train", name: str = "client", 
              exist_ok: bool = False):
        super().train(epochs, batch_size, imgsz, project, name, exist_ok)

    def train_and_send(self, manager, epochs=1):
        """
        训练模型并将 .pt 文件路径发送给管理器。

        参数:
            manager (FederationManager): 联邦学习管理器
            epochs (int): 训练的 epoch 数
        """
        self.train(epochs=epochs)
        manager.client_pt_paths.append(self.pt_path)

if __name__=="__main__":

    # 初始化服务器
    server = Server(
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/vd1.yaml",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        project="/root/autodl-tmp/serverruns/train",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )
    
    # 初始化客户端
    client1 = Client(
        server=server,
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/vd1.yaml",
        project="/root/autodl-tmp/client1runs/train",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )

    # 初始化客户端
    client2 = Client(
        server=server,
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/vd1.yaml",
        project="/root/autodl-tmp/client2runs/train",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )

    # 初始化客户端
    client3 = Client(
        server=server,
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/vd1.yaml",
        project="/root/autodl-tmp/client3runs/train",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )

    # 创建联邦学习管理器
    manager = FederationManager(server, [client1, client2, client3])

    # 模拟多轮训练和聚合
    for round in range(5):  # 进行 5 轮
        print(f"Round {round + 1}")
        # 清空上一轮的模型路径
        manager.client_pt_paths = []

        # 客户端训练并发送模型路径
        for client in manager.clients:
            client.train_and_send(manager, epochs=1)

        # 服务器聚合并分发
        manager.aggregate_and_distribute(server_trains=True)  # 服务器参与训练
        # 或者使用 manager.aggregate_and_distribute(server_trains=False)  # 服务器不训练 
