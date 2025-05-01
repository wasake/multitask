import torch
from ultralytics import YOLO
import os
import shutil
from cerberusdet.models.yolomanager import YOLOManager  # 假设 yolomanager.py 在此路径

class NetworkEndpoint():
    def __init__(self, pt_path: str, data_yaml: str, hyp: str, cfg: str, device: str = "cuda", project: str = "runs/trains"):
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
        self.prg = project
        self.device = device if torch.cuda.is_available() else "cpu"
        self.yolo_manager = YOLOManager(pt_path, device)
        self.optimizer_state = None  # 添加优化器状态存储

    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, project: str = "runs/train", 
              name: str = "exp", exist_ok: bool = False, resume_optimizer: bool = False):
        """
        训练模型。
        
        参数:
            epochs (int): 训练的 epoch 数
            batch_size (int): 批次大小
            imgsz (int): 图像尺寸
            project (str): 保存目录
            name (str): 实验名称
            exist_ok (bool): 是否覆盖现有目录
            resume_optimizer (bool): 是否恢复优化器状态
        """
        # 准备训练参数
        opt = {
            'weights': self.pt_path,
            'data': self.data_yaml,
            'hyp': self.hyp,
            'cfg': self.cfg,
            'epochs': epochs,
            'batch-size': batch_size,
            'imgsz': imgsz,
            'resume': False,
            'project': self.prg,
            'name': name,
            'exist-ok': exist_ok,
            'device': self.device,
            'optimizer_state': self.optimizer_state if resume_optimizer else None
        }
        
        # 调用修改后的训练函数
        from cerberusdet.train import run_with_optimizer_state
        new_pt_path, optimizer_state = run_with_optimizer_state(opt)
        self.pt_path = new_pt_path
        self.optimizer_state = optimizer_state  # 保存优化器状态

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

        # 聚合所有模型并获取更新后的 .pt 文件路径，但在内部已经实现该目标
        updated_pt_paths = self.server.aggregate(self.client_pt_paths, server_pt_path)

        # 分发聚合后的模型给对应的客户端
        # for i in range(len(self.clients)):
        #     self.clients[i].receive_pt(updated_pt_paths[i])
        #     print("检测通过")

class Server(NetworkEndpoint):
    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, 
              project: str = "/root/autodl-tmp/serverruns/train", name: str = "server", 
              exist_ok: bool = False, resume_optimizer: bool = False):
        super().train(epochs, batch_size, imgsz, project, name, exist_ok, resume_optimizer)

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

        for pt_path in client_pt_paths:
            # 临时更改 yolo_manager 的 pt_path 以加载客户端模型
            original_pt_path = self.yolo_manager.pt_path
            self.yolo_manager.pt_path = pt_path

            # 更新 backbone 参数
            self.yolo_manager.update_parameters(averaged_backbone, part="backbone")

            # 恢复原始 pt_path
            self.yolo_manager.pt_path = original_pt_path

        if server_pt_path:
            self.yolo_manager.update_parameters(averaged_backbone, part="backbone")
        
        return client_pt_paths

class Client(NetworkEndpoint):
    def train(self, epochs: int = 1, batch_size: int = 16, imgsz: int = 640, 
              project: str = "/root/autodl-fs/clientruns/train", name: str = "client", 
              exist_ok: bool = False, resume_optimizer: bool = False):
        super().train(epochs, batch_size, imgsz, project, name, exist_ok, resume_optimizer)

    def train_and_send(self, manager, epochs=1, round_idx=0):
        """
        训练模型并将 .pt 文件路径发送给管理器。

        参数:
            manager (FederationManager): 联邦学习管理器
            epochs (int): 训练的 epoch 数
            round_idx (int): 当前联邦学习轮次
        """
        # 第一轮不恢复优化器状态，后续轮次恢复
        resume_optimizer = (round_idx > 0) 
        self.train(epochs=epochs, resume_optimizer=resume_optimizer)
        manager.client_pt_paths.append(self.pt_path)

if __name__=="__main__":

    # 初始化服务器
    server = Server(
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/vd1.yaml",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        project="/root/autodl-fs/serverruns/train",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )
    
    # 初始化客户端
    client1 = Client(
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/UAV0.yaml",
        project="/root/autodl-fs/client1runswithserver/train",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )

    # 初始化客户端
    client2 = Client(
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/UAV1.yaml",
        project="/root/autodl-fs/client2runswithserver/train",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )

    # 初始化客户端
    client3 = Client(
        pt_path="pretrained/yolov8x_state_dict.pt",
        data_yaml="data/UAV2.yaml",
        project="/root/autodl-fs/client3runswithserver/train",
        hyp="data/hyps/hyp.cerber-voc_obj365.yaml",
        cfg="cerberusdet/models/yolov8x.yaml",
        device="0"
    )

    # 创建联邦学习管理器
    manager = FederationManager(server, [client1, client2, client3])

    # 模拟多轮训练和聚合
    for round in range(50):
        print(f"Round {round + 1}")
        manager.client_pt_paths = []

        # 客户端训练并发送模型路径（传递当前轮次）
        for client in manager.clients:
            client.train_and_send(manager, epochs=1, round_idx=round)

        # 服务器聚合并分发
        manager.aggregate_and_distribute(server_trains=False)
        
        print(f"Round {round + 1} completed")

    # mlruns占用大量内存，暂时不用这部分，将其删除 
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")
        print("已删除 mlruns 目录")