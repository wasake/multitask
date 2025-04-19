import socket
import torch
from ultralytics import YOLO
import os
import io
import time
from pathlib import Path


def client_training(client_id, client_socket, local_port, server_host="127.0.0.1", server_port=52323, max_rounds=5, timeout=60):
    # # 初始化YOLO模型
    global Path
    model =YOLO(r'datasets/yolov8m.pt')
    round_counter = 0  # 用来控制训练轮次
    # 动态生成保存目录，加入客户端ID和轮次信息
    project_dir = f'runs/detect/train{client_id}_{local_port}'  # 每个客户端有不同的目录，并包含训练轮次
    os.makedirs(project_dir, exist_ok=True)  # 创建目录（如果不存在）
    #
    while round_counter < max_rounds:
        round_counter += 1
        weights_echo = f'weights_epoch{round_counter}'
        print(f"客户端{client_id}开始第{round_counter}轮训练...")
        # 执行本地训练
        results = model.train(
            data=f'clientYaml/client{client_id}.yaml',  # 数据集配置文件
            epochs=1,  # 训练轮次
            batch=4,  # 批量大小
            imgsz=640,  # 图像大小
            save=True,  # 保存模型
            project=project_dir,  # 指定项目目录
            name=weights_echo,  # 指定子目录名称
            device='cpu',  # 如果没有GPU，使用 'cpu' 训练
            resume=False  # 是否从上次中断的地方继续训练
        )
        best_model_path = Path(project_dir) / weights_echo / 'weights' / 'best.pt'
        best_model_path = best_model_path.as_posix()  # 强制使用正斜杠
        print(f"客户端{client_id}开始上传模型...")
    #    best_model_path=r'datasets\WALDO30_yolov8m_640x640.pt'
        # 发送模型参数到服务器 (直接获取 state_dict)
        with open(best_model_path, 'rb') as file:
            model_data = torch.load(file)  # 加载模型
            if isinstance(model_data, dict) and 'model' in model_data:
                state_dict = model_data['model'].state_dict()  # 获取模型的参数（state_dict）

                # 使用 io.BytesIO 将状态字典保存为字节流
                buffer = io.BytesIO()
                torch.save(state_dict, buffer)
                buffer.seek(0)

                # 发送模型参数字节流到服务器
                _read_size = 102400  # 每次读取10KB数据
                data = buffer.read(_read_size)
                while data:
                    client_socket.send(data)  # 将读取的数据发送到服务器
                    data = buffer.read(_read_size)  # 继续读取下一部分数据

        # 发送文件结束标识符
        client_socket.send(b'fileEnd#')

        # 接收服务器的响应，告知客户端模型上传完成
        received_message = client_socket.recv(1024).decode()
        print(f"客户端{client_id}收到服务器响应：", received_message)

        # 判断是否继续训练（例如，服务器端发出停止信号）
        start_time = time.time()  # 记录开始时间
        while True:
            print(f"客户端{client_id}等待信号...")
            if "模型更新完成" in received_message:
                print(f"客户端{client_id}收到有效信号：{received_message}")
                break  # 如果信号中包含 "模型更新完成"，跳出循环，继续处理后续逻辑

            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"客户端{client_id}等待超时，没有接收到有效信号，结束训练")
                break  # 超过1分钟没有收到有效信号，退出循环

        # **在接收到“模型更新完成”的信号后，客户端开始接收全局模型**
        print(f"客户端{client_id}开始接收全局模型...")

        global_model_params = b''  # 用于存储从服务器接收的全局模型数据
        while True:
            data = client_socket.recv(102400)
            if not data:
                break
            global_model_params += data
            # 检查是否接收到文件结束标志
            if b'fileEnd#' in data:
                break  # 如果收到结束标志，退出循环

        # 使用 io.BytesIO 将接收到的字节流转换为文件对象
        buffer = io.BytesIO(global_model_params)
        try:
            global_model = torch.load(buffer)  # 使用 BytesIO 加载全局模型
            print("成功加载全局模型")
        except Exception as e:
            print(f"加载全局模型时发生错误: {e}")
            break

        try:
            model.model.load_state_dict(global_model.model.state_dict())  # 将全局模型的参数加载到客户端模型
            print(f"客户端{client_id}已更新全局模型，准备进行下一轮训练。")
        except Exception as e:
            print(f"更新客户端模型时发生错误: {e}")

        # 接收停止信号
        stop_signal = ""
        try:
            stop_signal = client_socket.recv(1024).decode()  # 等待服务器的停止信号
            print("接收的信号为:" + stop_signal)
        except socket.timeout:
            print("接收服务器信号超时")

        if stop_signal == "训练结束":
            print("客户端收到训练结束信号，停止训练")
            break  # 跳出循环，结束客户端进程

    # 保持连接直到服务器发出结束信号
    print(f"客户端{client_id}训练结束，关闭连接")
    client_socket.close()


# 启动客户端的训练与上传
if __name__ == "__main__":
    client_id = 1  # 假设启动客户端1
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('192.168.1.107', 10021))  # 在外部建立连接 IP Port
    local_address, local_port = client_socket.getsockname()
    print(f"客户端{client_id}的本地端口号为: {local_port}")
    client_training(client_id, client_socket,local_port)  # 传递已建立的socket连接
