import socket
import threading
import time
import torch
import os
import io
from ultralytics import YOLO
from threading import Lock  # 导入锁

def average_models(models):
    model_params = [model.state_dict() for model in models]
    averaged_params = {}

    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float32)  # 转换为32位浮动点数
        averaged_params[param_name] = torch.mean(params, dim=0)

    return averaged_params  # 返回聚合后的全局模型参数

def handle_client(client_socket, addr, models, global_model, client_sockets, aggregation_rounds, lock):
    # 聚合轮次控制
    aggregation_round = 0
    max_rounds = aggregation_rounds
    while aggregation_round < max_rounds:
        aggregation_round += 1
        _read_size = 102400  # 每次接收的块大小为100KB

        temp_folder_name = f"received_model/{addr[0]}_{addr[1]}"
        if not os.path.exists(temp_folder_name):
            os.makedirs(temp_folder_name)

        received_data = b""
        while True:
            data = client_socket.recv(_read_size)
            if b'fileEnd#' in data:  # 判断文件结束标志
                received_data += data[:-8]  # 去除文件结束标志
                break
            received_data += data  # 持续接收数据

        buffer = io.BytesIO(received_data)
        pt_file_path = f"{temp_folder_name}/yolo_epoch{aggregation_round}.pt"
        with open(pt_file_path, "wb") as f:
            f.write(received_data)
        print(f"保存模型文件到: {pt_file_path}")

        try:
            model_data = torch.load(buffer)
            model = YOLO(r'datasets/yolov8m.pt')  # 加载YOLO模型
            # model = YOLO('./yolov8s.yaml')  # 加载YOLO配置文件
            model.model.load_state_dict(model_data)  # 加载客户端模型参数
            models.append(model)  # 将客户端模型添加到模型列表
        except Exception as e:
            print(f"错误加载客户端{addr}模型: {e}")

        if len(models) == 2:  # 假设我们有2个客户端上传了模型
            try:
                lock.acquire()  # 获取锁，防止多个线程同时更新全局模型
                ensemble_model_params = average_models(models)
                global_model.load_state_dict(ensemble_model_params)  # 更新全局模型参数

                buffer = io.BytesIO()
                torch.save(global_model, buffer)
                buffer.seek(0)

                for client in client_sockets:  # 遍历所有客户端连接
                    try:
                        if client.fileno() != -1:  # 检查套接字是否有效
                            client.send("模型更新完成".encode('utf-8'))
                            buffer.seek(0)  # 重置指针到开头
                            client.sendall(buffer.read())
                            client.send(b'fileEnd#')  # 确保客户端知道数据发送完毕
                    except Exception as e:
                        print(f"发送全局模型到客户端{client.getpeername()}时发生错误: {e}")
                        client_sockets.remove(client)  # 移除失效客户端

                # 向所有客户端发送“开始下一轮”信号
                for client in client_sockets:
                    try:
                        if client.fileno() != -1:  # 检查套接字是否有效
                            print("开始下一轮")
                            time.sleep(10)  # 每次等待信号时延迟2秒，避免过度占用CPU
                            client.send("开始下一轮训练".encode('utf-8'))
                    except Exception as e:
                        print(f"发送信号到客户端{client.getpeername()}时发生错误: {e}")
                        client_sockets.remove(client)  # 移除失效客户端
            except Exception as e:
                print(f"模型聚合时发生错误: {e}")
            finally:
                lock.release()  # 释放锁，允许其他线程执行

            models.clear()  # 重置模型列表

    # 保存最终的模型
    final_model_path = "final_global_model.pt"
    torch.save(global_model, final_model_path)
    print(f"训练完成，最终全局模型已保存为: {final_model_path}")
    return

# 服务器监听部分
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '127.0.0.1'  # 本机IP地址
port = 52323  # 端口号

server_socket.bind((host, port))
server_socket.listen(5)

print("等待客户端连接...")

models = []  # 用于保存客户端上传的模型
global_model = YOLO(r'datasets/yolov8m.pt')  # 加载YOLO模型

client_sockets = []  # 存储所有客户端的连接
aggregation_rounds = 5  # 聚合的最大轮次

# 创建锁
lock = Lock()

# 无限循环，等待客户端连接
while True:
    client_socket, addr = server_socket.accept()
    print(f"连接地址: {addr}")
    client_sockets.append(client_socket)  # 将客户端套接字添加到列表
    client_thread = threading.Thread(target=handle_client,
                                     args=(client_socket, addr, models, global_model, client_sockets, aggregation_rounds, lock))
    client_thread.start()  # 启动新线程处理每个客户端
