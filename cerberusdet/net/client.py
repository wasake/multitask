import socket
import pickle
import logging
from cerberusdet.models.yolomanager import YOLOManager
from typing import Dict, Any

class Client:
    def __init__(
        self,
        client_id: int,
        server_host: str = "127.0.0.1",
        server_port: int = 5000
    ):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.yolo_manager = None
        self.config = None
        self.rounds = None
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def connect_to_server(self):
        """连接到服务器"""
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.server_host, self.server_port))
        logging.info(f"客户端 {self.client_id} 已连接到服务器")

    def receive_data(self):
        """接收服务器数据"""
        try:
            # 首先接收数据大小
            size_data = self.conn.recv(8)
            size = int.from_bytes(size_data, byteorder='big')
            
            # 分块接收数据
            data = bytearray()
            while len(data) < size:
                chunk = self.conn.recv(min(4096, size - len(data)))
                if not chunk:
                    raise ConnectionError("连接中断")
                data.extend(chunk)
                
            return pickle.loads(data)
        except Exception as e:
            logging.error(f"接收数据失败: {e}")
            return None

    def send_data(self, data):
        """发送数据到服务器"""
        try:
            # 序列化数据
            serialized_data = pickle.dumps(data)
            # 先发送数据大小
            size = len(serialized_data)
            self.conn.sendall(size.to_bytes(8, byteorder='big'))
            # 分块发送数据
            chunk_size = 4096
            for i in range(0, len(serialized_data), chunk_size):
                chunk = serialized_data[i:i + chunk_size]
                self.conn.sendall(chunk)
        except Exception as e:
            logging.error(f"发送数据失败: {e}")

    def receive_initial_data(self):
        """接收基础模型参数和配置"""
        data = self.receive_data()
        if data:
            self.yolo_manager = YOLOManager("temp.pt")  # 创建临时模型文件
            self.yolo_manager.update_parameters(data["base_model_params"])
            self.config = data["config"]
            self.rounds = data["rounds"]
            logging.info(f"客户端 {self.client_id} 初始化完成")

    def train_and_send(self):
        """训练模型并发送模型参数"""
        for round_num in range(self.rounds):
            logging.info(f"客户端 {self.client_id} 开始第 {round_num + 1} 轮训练")

            # 模型训练
            self.yolo_manager.train(
                data_yaml=self.config["data_yaml"],
                hyp=self.config["hyp"],
                cfg=self.config["cfg"],
                epochs=1,
                batch_size=self.config["batch_size"],
                imgsz=self.config["imgsz"],
                project=self.config["project"],
                name=f"client_{self.client_id}_round_{round_num}"
            )

            # 发送模型参数
            model_params = self.yolo_manager.get_parameters("all")
            self.send_data({"model_params": model_params})

            # 接收聚合后的模型参数
            data = self.receive_data()
            if data and "aggregated_model_params" in data:
                self.yolo_manager.update_parameters(data["aggregated_model_params"])
                logging.info(f"客户端 {self.client_id} 更新聚合模型完成")

    def start(self):
        """启动客户端"""
        self.connect_to_server()
        self.receive_initial_data()
        self.train_and_send()


if __name__ == "__main__":
    # 创建并启动客户端示例
    client = Client(
        client_id=1,
        server_host="192.168.1.1",
        server_port=5000
    )
    client.start()