import socket
import pickle
import logging
import torch
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
        self.optimizer_state = None  # 优化器状态存储
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def connect_to_server(self):
        """连接到服务器"""
        try:
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.connect((self.server_host, self.server_port))
            logging.info(f"客户端 {self.client_id} 已连接到服务器 {self.server_host}:{self.server_port}")
        except Exception as e:
            logging.error(f"连接服务器失败: {e}")
            raise

    def receive_data(self):
        """接收服务器数据"""
        try:
            # 首先接收数据大小
            size_data = self.conn.recv(8)
            if not size_data:
                raise ConnectionError("接收数据大小失败，连接可能已关闭")
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
            logging.debug(f"客户端 {self.client_id} 已发送 {size} 字节数据")
        except Exception as e:
            logging.error(f"发送数据失败: {e}")
            raise

    def receive_initial_data(self):
        """接收基础模型参数和配置"""
        logging.info(f"客户端 {self.client_id} 等待接收初始数据...")
        data = self.receive_data()
        if data:
            self.yolo_manager = YOLOManager("temp.pt")  # 创建临时模型文件
            self.yolo_manager.update_parameters(data["base_model_params"])
            self.config = data["config"]
            self.rounds = data["rounds"]
            logging.info(f"客户端 {self.client_id} 初始化完成，将训练 {self.rounds} 轮")
        else:
            logging.error("初始化数据接收失败")
            raise RuntimeError("无法接收初始化数据")

    def train_model(self, round_num, resume_optimizer=False):
        """训练模型并保存优化器状态"""
        # 准备训练参数
        opt = {
            'weights': self.yolo_manager.pt_path,
            'data': self.config["data_yaml"],
            'hyp': self.config["hyp"],
            'cfg': self.config["cfg"],
            'epochs': 1,
            'batch-size': self.config["batch_size"],
            'imgsz': self.config["imgsz"],
            'resume': False,
            'project': self.config["project"],
            'name': f"client_{self.client_id}_round_{round_num}",
            'exist-ok': True,
            'device': '0',  # 假设使用第一个GPU
            'optimizer_state': self.optimizer_state if resume_optimizer else None
        }
        
        logging.info(f"客户端 {self.client_id} 开始第 {round_num + 1} 轮训练，{'恢复优化器状态' if resume_optimizer else '初始化新优化器'}")
        
        # 调用训练函数
        try:
            from cerberusdet.train import run_with_optimizer_state
            new_pt_path, new_optimizer_state = run_with_optimizer_state(opt)
            self.yolo_manager.pt_path = new_pt_path
            self.optimizer_state = new_optimizer_state
            logging.info(f"客户端 {self.client_id} 完成第 {round_num + 1} 轮训练，更新优化器状态，保存模型至 {new_pt_path}")
        except Exception as e:
            logging.error(f"训练过程出错: {e}")
            raise

    def train_and_send(self):
        """训练模型并发送模型参数"""
        for round_num in range(self.rounds):
            logging.info(f"===== 客户端 {self.client_id} 开始第 {round_num + 1}/{self.rounds} 轮训练 =====")

            try:
                # 模型训练，第一轮不恢复优化器，后续轮次恢复
                resume_optimizer = (round_num > 0)
                self.train_model(round_num, resume_optimizer)

                # 发送模型参数
                logging.info(f"客户端 {self.client_id} 正在发送模型参数...")
                model_params = self.yolo_manager.get_parameters("all")
                self.send_data({"model_params": model_params})

                # 接收聚合后的模型参数
                logging.info(f"客户端 {self.client_id} 等待接收聚合后的模型参数...")
                data = self.receive_data()
                if data and "aggregated_model_params" in data:
                    self.yolo_manager.update_parameters(data["aggregated_model_params"], part="backbone")
                    logging.info(f"客户端 {self.client_id} 更新聚合模型backbone部分完成")
                else:
                    logging.error(f"客户端 {self.client_id} 未能接收到聚合模型参数")
            except Exception as e:
                logging.error(f"客户端 {self.client_id} 在第 {round_num + 1} 轮训练中遇到错误: {e}")
                # 尝试继续下一轮训练而不崩溃
                continue

            logging.info(f"===== 客户端 {self.client_id} 完成第 {round_num + 1}/{self.rounds} 轮训练 =====")

    def start(self):
        """启动客户端"""
        try:
            self.connect_to_server()
            self.receive_initial_data()
            self.train_and_send()
            logging.info(f"客户端 {self.client_id} 完成所有训练轮次")
        except Exception as e:
            logging.error(f"客户端 {self.client_id} 运行出错: {e}")
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
                logging.info(f"客户端 {self.client_id} 已关闭连接")


if __name__ == "__main__":
    # 创建并启动客户端示例
    client = Client(
        client_id=1,
        server_host="192.168.1.1",
        server_port=5000
    )
    client.start()