import socket
import os
import pickle
import logging
from cerberusdet.models.yolomanager import YOLOManager
from typing import List, Dict, Any

class Server:
    def __init__(
        self,
        base_model_path: str,
        num_clients: int,
        server_config: Dict[str, Any],
        client_configs: List[Dict[str, Any]],
        host: str = "127.0.0.1",
        port: int = 5000,
        rounds: int = 50
    ):
        self.base_model_path = base_model_path
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.rounds = rounds
        self.clients = []
        
        self.server_config = server_config
        self.client_configs = client_configs
        
        self.yolo_manager = YOLOManager(base_model_path)
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def start(self):
        """启动服务器"""
        logging.info("服务器启动，等待客户端连接...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.num_clients)

            for i in range(self.num_clients):
                conn, addr = server_socket.accept()
                logging.info(f"客户端 {addr} 已连接")
                self.clients.append(conn)

                # 获取基础模型参数
                base_model_params = self.yolo_manager.get_parameters("all")
                
                # 发送模型参数和配置
                self.send_data(conn, {
                    "base_model_params": base_model_params,  # 发送完整模型参数
                    "config": self.client_configs[i],
                    "rounds": self.rounds
                })

            # 开始训练和聚合
            self.train_and_aggregate()

    def send_data(self, conn, data):
        """发送数据到客户端"""
        try:
            # 序列化数据
            serialized_data = pickle.dumps(data)
            # 先发送数据大小
            size = len(serialized_data)
            conn.sendall(size.to_bytes(8, byteorder='big'))
            # 分块发送数据
            chunk_size = 4096
            for i in range(0, len(serialized_data), chunk_size):
                chunk = serialized_data[i:i + chunk_size]
                conn.sendall(chunk)
        except Exception as e:
            logging.error(f"发送数据失败: {e}")

    def receive_data(self, conn):
        """接收客户端数据"""
        try:
            # 首先接收数据大小
            size_data = conn.recv(8)
            size = int.from_bytes(size_data, byteorder='big')
            
            # 分块接收数据
            data = bytearray()
            while len(data) < size:
                chunk = conn.recv(min(4096, size - len(data)))
                if not chunk:
                    raise ConnectionError("连接中断")
                data.extend(chunk)
                
            return pickle.loads(data)
        except Exception as e:
            logging.error(f"接收数据失败: {e}")
            return None

    def train_and_aggregate(self):
        """训练和聚合模型"""
        for round_num in range(self.rounds):
            logging.info(f"开始第 {round_num + 1} 轮训练")

            # 接收客户端模型参数
            client_params = []
            for conn in self.clients:
                data = self.receive_data(conn)
                if data:
                    client_params.append(data["model_params"])

            # 聚合模型
            aggregated_params = self.aggregate_models(client_params)

            # 发送聚合后的模型参数给客户端
            for conn in self.clients:
                self.send_data(conn, {"aggregated_model_params": aggregated_params})

    def aggregate_models(self, client_params):
        """聚合客户端模型参数"""
        client_backbones = [params for params in client_params]
        averaged_backbone = {}
        for key in client_backbones[0].keys():
            averaged_backbone[key] = sum([backbone[key] for backbone in client_backbones]) / len(client_backbones)

        # 更新服务器模型
        self.yolo_manager.update_parameters(averaged_backbone, part="backbone")
        logging.info("模型聚合完成")
        return averaged_backbone


if __name__ == "__main__":
    # 服务器配置示例
    server_config = {
        "data_yaml": "data/vd1.yaml",
        "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
        "cfg": "cerberusdet/models/yolov8x.yaml",
        "project": "/root/autodl-fs/serverruns/train"
    }

    # 客户端配置示例
    client_configs = [
        {
            "data_yaml": "data/UAV0.yaml",
            "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
            "cfg": "cerberusdet/models/yolov8x.yaml",
            "project": "/root/autodl-fs/client1runs/train",
            "batch_size": 16,
            "imgsz": 640
        },
        {
            "data_yaml": "data/UAV1.yaml",
            "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
            "cfg": "cerberusdet/models/yolov8x.yaml",
            "project": "/root/autodl-fs/client2runs/train",
            "batch_size": 16,
            "imgsz": 640
        },
        {
            "data_yaml": "data/UAV2.yaml",
            "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
            "cfg": "cerberusdet/models/yolov8x.yaml",
            "project": "/root/autodl-fs/client3runs/train",
            "batch_size": 16,
            "imgsz": 640
        }
    ]

    server = Server(
        base_model_path="pretrained/yolov8x_state_dict.pt",
        num_clients=3,
        server_config=server_config,
        client_configs=client_configs,
        host="192.168.1.0",
        port=5000,
        rounds=50
    )
    server.start()