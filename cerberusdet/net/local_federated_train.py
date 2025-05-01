import threading
import queue
import logging
import time
from typing import Dict, Any, List
from cerberusdet.models.yolomanager import YOLOManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s"
)

# 共享消息队列，模拟网络通信
client_to_server_queue = queue.Queue()
server_to_client_queues = {}

class ServerThread(threading.Thread):
    """服务器线程，负责接收客户端参数并进行聚合"""
    
    def __init__(
        self,
        base_model_path: str,
        server_config: Dict[str, Any],
        client_configs: List[Dict[str, Any]],
        rounds: int = 50,
        server_trains: bool = True
    ):
        super().__init__(name="Server")
        self.base_model_path = base_model_path
        self.server_config = server_config
        self.client_configs = client_configs
        self.rounds = rounds
        self.server_trains = server_trains
        self.client_count = len(client_configs)
        self.yolo_manager = YOLOManager(base_model_path)
        self.optimizer_state = None
        
        logging.info(f"服务器初始化完成，训练模式: {'开启' if server_trains else '关闭'}")

    def train_model(self, round_num, resume_optimizer=False):
        """训练服务器模型并保存优化器状态"""
        if not self.server_trains:
            logging.info("服务器训练已禁用，跳过训练步骤")
            return None
            
        # 准备训练参数
        batch_size = self.server_config.get("batch_size", 16)
        imgsz = self.server_config.get("imgsz", 640)
        
        opt = {
            'weights': self.yolo_manager.pt_path,
            'data': self.server_config["data_yaml"],
            'hyp': self.server_config["hyp"],
            'cfg': self.server_config["cfg"],
            'epochs': 1,
            'batch-size': batch_size,
            'imgsz': imgsz,
            'resume': False,
            'project': self.server_config["project"],
            'name': f"server_round_{round_num}",
            'exist-ok': True,
            'device': '0',
            'optimizer_state': self.optimizer_state if resume_optimizer else None
        }
        
        logging.info(f"服务器开始第 {round_num + 1} 轮训练，{'恢复优化器状态' if resume_optimizer else '初始化新优化器'}")
        
        # 调用训练函数
        from cerberusdet.train import run_with_optimizer_state
        new_pt_path, new_optimizer_state = run_with_optimizer_state(opt)
        self.yolo_manager.pt_path = new_pt_path
        self.optimizer_state = new_optimizer_state
        logging.info(f"服务器完成第 {round_num + 1} 轮训练，更新优化器状态，保存模型至 {new_pt_path}")
        
        return self.yolo_manager.get_parameters("all")
        
    def aggregate_models(self, client_params, server_params=None):
        """聚合客户端和服务器模型参数（仅backbone部分）"""
        # 收集所有参与聚合的参数
        all_params = client_params.copy()
        if server_params:
            all_params.append(server_params)
            logging.info(f"聚合 {len(client_params)} 个客户端模型和 1 个服务器模型的backbone部分")
        else:
            logging.info(f"聚合 {len(client_params)} 个客户端模型的backbone部分")
        
        # 只选择backbone部分的参数进行聚合
        backbone_keys = [k for k in all_params[0].keys() if "backbone" in k]
        averaged_params = {}
        
        # 只对backbone部分的参数进行平均
        for key in backbone_keys:
            averaged_params[key] = sum([params[key] for params in all_params]) / len(all_params)
        
        # 更新服务器模型backbone部分
        self.yolo_manager.update_parameters(averaged_params, part="backbone")
        logging.info("模型backbone聚合完成，服务器模型已更新")
        return averaged_params

    def run(self):
        """启动服务器线程的主循环"""
        # 准备初始模型参数
        base_model_params = self.yolo_manager.get_parameters("all")
        
        # 向所有客户端发送初始模型和配置
        for i in range(self.client_count):
            server_to_client_queues[i].put({
                "type": "init",
                "base_model_params": base_model_params,
                "config": self.client_configs[i],
                "rounds": self.rounds
            })
            logging.info(f"服务器向客户端 {i+1} 发送初始数据")
        
        # 训练和聚合循环
        for round_num in range(self.rounds):
            logging.info(f"===== 开始第 {round_num + 1}/{self.rounds} 轮联邦学习 =====")
            
            # 服务器训练（如果启用）
            server_params = None
            if self.server_trains:
                resume_optimizer = (round_num > 0)
                server_params = self.train_model(round_num, resume_optimizer)
            
            # 接收客户端模型参数
            client_params = []
            received_count = 0
            
            while received_count < self.client_count:
                try:
                    data = client_to_server_queue.get(timeout=300)  # 设置超时时间为5分钟
                    if data and "client_id" in data and "model_params" in data:
                        client_id = data["client_id"]
                        client_params.append(data["model_params"])
                        logging.info(f"服务器收到客户端 {client_id} 的模型参数")
                        received_count += 1
                except queue.Empty:
                    logging.error("等待客户端参数超时，继续处理已接收的参数")
                    break
            
            if not client_params:
                logging.error("没有接收到任何客户端模型参数，跳过聚合")
                continue
                
            # 聚合模型
            logging.info("开始聚合模型参数...")
            aggregated_params = self.aggregate_models(client_params, server_params)
            
            # 向所有客户端发送聚合后的参数
            for i in range(self.client_count):
                server_to_client_queues[i].put({
                    "type": "update",
                    "aggregated_model_params": aggregated_params
                })
                logging.info(f"服务器向客户端 {i+1} 发送聚合后的模型参数")
                
            logging.info(f"===== 完成第 {round_num + 1}/{self.rounds} 轮联邦学习 =====")
        
        # 训练结束，通知所有客户端
        for i in range(self.client_count):
            server_to_client_queues[i].put({"type": "done"})
        logging.info("服务器训练完成，已通知所有客户端")


class ClientThread(threading.Thread):
    """客户端线程，负责训练模型并发送参数给服务器"""
    
    def __init__(self, client_id: int):
        super().__init__(name=f"Client-{client_id}")
        self.client_id = client_id
        self.yolo_manager = None
        self.config = None
        self.rounds = None
        self.optimizer_state = None
        server_to_client_queues[client_id] = queue.Queue()  # 创建服务器到该客户端的消息队列
    
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
            'device': '0',  # 使用第一个GPU
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

    def run(self):
        """启动客户端线程的主循环"""
        # 等待接收初始化数据
        logging.info(f"客户端 {self.client_id} 等待接收初始数据...")
        init_data = server_to_client_queues[self.client_id].get()
        
        if init_data["type"] == "init":
            self.yolo_manager = YOLOManager("temp.pt")  # 创建临时模型文件
            self.yolo_manager.update_parameters(init_data["base_model_params"])
            self.config = init_data["config"]
            self.rounds = init_data["rounds"]
            logging.info(f"客户端 {self.client_id} 初始化完成，将训练 {self.rounds} 轮")
        else:
            logging.error("初始化数据接收失败")
            return
            
        # 开始训练和参数交换循环
        for round_num in range(self.rounds):
            logging.info(f"===== 客户端 {self.client_id} 开始第 {round_num + 1}/{self.rounds} 轮训练 =====")
            
            try:
                # 模型训练，第一轮不恢复优化器，后续轮次恢复
                resume_optimizer = (round_num > 0)
                self.train_model(round_num, resume_optimizer)
                
                # 发送模型参数到服务器
                logging.info(f"客户端 {self.client_id} 正在发送模型参数...")
                model_params = self.yolo_manager.get_parameters("all")
                client_to_server_queue.put({
                    "client_id": self.client_id,
                    "model_params": model_params
                })
                
                # 接收聚合后的模型参数
                logging.info(f"客户端 {self.client_id} 等待接收聚合后的模型参数...")
                update_data = server_to_client_queues[self.client_id].get()
                
                if update_data["type"] == "update" and "aggregated_model_params" in update_data:
                    self.yolo_manager.update_parameters(update_data["aggregated_model_params"], part="backbone")
                    logging.info(f"客户端 {self.client_id} 更新聚合模型backbone部分完成")
                else:
                    logging.error(f"客户端 {self.client_id} 未能接收到聚合模型参数")
                    
            except Exception as e:
                logging.error(f"客户端 {self.client_id} 在第 {round_num + 1} 轮训练中遇到错误: {e}")
                # 尝试继续下一轮训练而不崩溃
                continue
                
            logging.info(f"===== 客户端 {self.client_id} 完成第 {round_num + 1}/{self.rounds} 轮训练 =====")
            
        # 等待最终通知
        final_msg = server_to_client_queues[self.client_id].get()
        if final_msg["type"] == "done":
            logging.info(f"客户端 {self.client_id} 已完成所有训练轮次")


def main():
    """主函数，启动本地联邦学习训练"""
    # 服务器配置
    server_config = {
        "data_yaml": "data/vd1.yaml",
        "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
        "cfg": "cerberusdet/models/yolov8x.yaml",
        "project": "runs/server_federated",
        "batch_size": 16,
        "imgsz": 640
    }

    # 客户端配置
    client_configs = [
        {
            "data_yaml": "data/UAV0.yaml",
            "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
            "cfg": "cerberusdet/models/yolov8x.yaml",
            "project": "runs/client1_federated",
            "batch_size": 16,
            "imgsz": 640
        },
        {
            "data_yaml": "data/UAV1.yaml",
            "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
            "cfg": "cerberusdet/models/yolov8x.yaml",
            "project": "runs/client2_federated",
            "batch_size": 16,
            "imgsz": 640
        },
        {
            "data_yaml": "data/UAV2.yaml",
            "hyp": "data/hyps/hyp.cerber-voc_obj365.yaml",
            "cfg": "cerberusdet/models/yolov8x.yaml",
            "project": "runs/client3_federated",
            "batch_size": 16,
            "imgsz": 640
        }
    ]
    
    num_clients = len(client_configs)
    rounds = 10  # 训练轮次，可根据需要调整
    base_model_path = "pretrained/yolov8x_state_dict.pt"
    server_trains = True  # 服务器是否参与训练
    
    # 创建并启动服务器线程
    server_thread = ServerThread(
        base_model_path=base_model_path,
        server_config=server_config,
        client_configs=client_configs,
        rounds=rounds,
        server_trains=server_trains
    )
    
    # 创建并启动客户端线程
    client_threads = []
    for i in range(num_clients):
        client_thread = ClientThread(client_id=i)
        client_threads.append(client_thread)
    
    # 启动所有线程
    server_thread.start()
    time.sleep(2)  # 确保服务器先启动
    for client in client_threads:
        client.start()
        time.sleep(1)  # 避免客户端同时启动
    
    # 等待所有线程完成
    server_thread.join()
    for client in client_threads:
        client.join()
        
    logging.info("本地联邦学习训练完成！")


if __name__ == "__main__":
    main()