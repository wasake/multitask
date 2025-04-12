import torch
import torch.nn as nn
import numpy as np

from ultralytics import YOLO

class Block:
    """管理网络块的元数据，记录父子关系和任务信息"""
    def __init__(self, index, module, task_id=None):
        self.index = index          # 块的唯一索引
        self.module = module        # 对应的 nn.Module
        self.task_id = task_id      # 任务 ID（仅 head 块有值）
        self.parent_index = None    # 父块索引
        self.children_indices = []  # 子块索引列表

    def stack_on(self, parent_block):
        """将当前块连接到父块"""
        self.parent_index = parent_block.index
        parent_block.children_indices.append(self.index)
        return self

class BlockController(nn.Module):
    def __init__(self, task_ids, nc, verbose=True):
        """
        初始化 BlockController，用于共享 backbone 和分支 neck/head。

        参数:
            task_ids (list): 任务 ID 列表，例如 ['task1', 'task2']
            nc (int or list): 类别数量，支持整数或列表
            verbose (bool): 是否打印详细日志
        """
        super().__init__()
        self.task_ids = task_ids
        self.verbose = verbose
        self.blocks = []           # 存储 Block 对象
        self.head_indices = {}     # 任务 ID 到 head 块索引的映射
        self.execution_plans = {}  # 每个任务的执行顺序

        # 处理 nc 格式
        if isinstance(nc, int):
            self.nc = [nc] * len(task_ids)  # 为每个任务分配相同的 nc
        elif isinstance(nc, list):
            assert len(nc) == len(task_ids), f"nc 列表长度必须与任务数量一致，得到 {nc}"
            self.nc = nc
        else:
            raise ValueError(f"nc 必须是整数或列表，得到 {type(nc)}")

    def add_block(self, module, parent_block=None):
        """
        添加一个网络块。

        参数:
            module (nn.Module): 要添加的网络模块
            parent_block (Block, optional): 父块

        返回:
            Block: 新添加的块对象
        """
        index = len(self.blocks)
        block = Block(index, module)
        self.blocks.append(block)
        if parent_block is not None:
            block.stack_on(parent_block)
        if self.verbose:
            print(f"Added block {index}: {type(module).__name__}")
        return block

    def add_head(self, module, task_id, parent_block=None):
        """
        添加任务头部块。

        参数:
            module (nn.Module): 头部模块
            task_id (str): 任务 ID
            parent_block (Block, optional): 父块

        返回:
            Block: 新添加的头部块
        """
        assert task_id in self.task_ids, f"未知的任务 ID: {task_id}"
        head_block = self.add_block(module, parent_block)
        head_block.task_id = task_id
        self.head_indices[task_id] = head_block.index
        if self.verbose:
            print(f"Added head for {task_id} at index {head_block.index}")
        return head_block

    def load_and_split_yolo_models(self, pt_files):
        """
        加载多个 YOLO 模型，拼接为共享 backbone 和分支 neck/head。

        参数:
            pt_files (list): .pt 文件路径列表
        """
        assert len(pt_files) == len(self.task_ids), "任务数量与模型数量必须一致"

        # 解决 NumPy 反序列化问题
        torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])

        # 加载 YOLO 模型
        yolo_models = [YOLO(pt_file) for pt_file in pt_files]
        models = [yolo.model for yolo in yolo_models]  # 获取 nn.Module

        # # 检查 backbone 一致性
        # backbone_state_dict = models[0]['model'].state_dict()
        # for i, model in enumerate(models[1:], 1):
        #     current_state_dict = model['model'].state_dict()
        #     assert set(backbone_state_dict.keys()) == set(current_state_dict.keys()), f"模型 {i} 的 state_dict 键不一致"
        #     for key in backbone_state_dict:
        #         assert torch.equal(backbone_state_dict[key], current_state_dict[key]), f"模型 {i} 的 backbone 参数 {key} 不一致"

        # 获取模型层
        model = models[0]
        model_layers = list(model.children())

        # 如果顶层是 ModuleList，展开子模块
        if len(model_layers) == 1 and isinstance(model_layers[0], nn.ModuleList):
            if self.verbose:
                print("顶层是 ModuleList，展开子模块")
            model_layers = list(model_layers[0])

        # 打印层结构以调试
        if self.verbose:
            for i, layer in enumerate(model_layers):
                print(f"Layer {i}: {type(layer).__name__}")

        # 调整层划分（针对 YOLOv8x）
        N = 1   # backbone 层数（前 9 层：Conv、C2f）,实际backbone只有1层sequential
        M = 12  # neck 层数（10-21 层：SPPF、Upsample、Concat）

        # 提取共享 backbone
        backbone_layers = model_layers[0]
        backbone = nn.Sequential(backbone_layers)
        backbone_block = self.add_block(backbone)
        self.register_module(f"block_{backbone_block.index}", backbone)

        # 为每个任务添加 neck 和 head
        for task_id, model in zip(self.task_ids, models):
            
            concat_inputs = {
                15: [6, 14],  # Concat: backbone 第 6 层 + neck 第 14 层
                18: [4, 17],  # Concat: backbone 第 4 层 + neck 第 17 层
                21: [8, 20],  # Concat: backbone 第 8 层 + neck 第 20 层
            }
            
            # 这里要拆分成为layers
            neck_layers = list(model.children())[0][N:N+M]
            head_layers = list(model.children())[0][N+M:]

            for i in 
            neck = nn.Sequential(*neck_layers)
            head = nn.Sequential(*head_layers)

            # 添加 neck 和 head
            neck_block = self.add_block(neck, backbone_block)
            head_block = self.add_head(head, task_id, neck_block)

            # 注册模块以确保参数可训练
            self.register_module(f"block_{neck_block.index}", neck)
            self.register_module(f"block_{head_block.index}", head)

        # 构建执行计划
        self.build_execution_plans()

        for block in self.blocks:
            print(f"Block {block.index}: parent_index={block.parent_index}, task_id={block.task_id}")

    def build_execution_plans(self):
        """
        为每个任务生成执行计划（块索引序列）。
        """
        for task_id in self.task_ids:
            head_index = self.head_indices[task_id]
            execution_order = []
            current_block = self.blocks[head_index]

            # 从 head 向上回溯到 backbone
            while current_block is not None:
                execution_order.append(current_block.index)
                if current_block.parent_index is not None:
                    current_block = self.blocks[current_block.parent_index]
                else:
                    current_block = None

            # 反转顺序，从 backbone 到 head
            execution_order = execution_order[::-1]
            self.execution_plans[task_id] = execution_order
            if self.verbose:
                print(f"Execution plan for {task_id}: {execution_order}")

    def forward_task(self, x, task_id):
        """
        执行指定任务的前向传播。

        参数:
            x (Tensor): 输入张量
            task_id (str): 任务 ID

        返回:
            Tensor: 任务输出
        """
        if task_id not in self.execution_plans:
            raise ValueError(f"未知的任务 ID: {task_id}")

        execution_order = self.execution_plans[task_id]
        outputs = {}

        for index in execution_order:
            block = self.blocks[index]
            parent_index = block.parent_index

            if parent_index is None:
                input_tensor = x
            else:
                input_tensor = outputs.get(parent_index)
                if input_tensor is None:
                    raise ValueError(f"Block {index} 的输入为 None，父块索引={parent_index}")

            if self.verbose:
                print(f"Processing block {index} ({type(block.module).__name__}) with input shape {input_tensor.shape if hasattr(input_tensor, 'shape') else 'None'}")

            # 执行前向传播
            output = block.module(input_tensor)

            # 处理可能的列表输出（例如 Detect 层）
            if isinstance(output, (list, tuple)):
                valid_outputs = [o for o in output if o is not None]
                if not valid_outputs:
                    raise ValueError(f"Block {index} 返回了空列表: {output}")
                output = valid_outputs  # 取所有张量！！！
                if self.verbose:
                    print(f"Block {index} returned a list")

            # 检查 None 输出
            if output is None:
                raise ValueError(f"Block {index} 返回了 None，类型={type(block.module)}")

            outputs[index] = output

            if block.task_id == task_id:
                return output

        raise ValueError(f"未找到任务 {task_id} 的输出")

    def forward(self, x, task_id=None):
        """
        前向传播，支持指定任务或运行所有任务。

        参数:
            x (Tensor): 输入张量
            task_id (str, optional): 任务 ID，若为 None 则运行所有任务

        返回:
            dict 或 Tensor: 所有任务的输出字典，或指定任务的输出
        """
        if task_id is not None:
            return self.forward_task(x, task_id)
        else:
            return {tid: self.forward_task(x, tid) for tid in self.task_ids}

# 示例运行
if __name__ == "__main__":
    task_ids = ['task1', 'task2']
    pt_files = ['pretrained/client_model.pt', 'pretrained/client_model.pt']
    model = BlockController(task_ids=task_ids, nc=[20, 20], verbose=True)
    model.load_and_split_yolo_models(pt_files)

    # 测试前向传播
    input_tensor = torch.randn(1, 3, 640, 640)
    outputs = model(input_tensor)
    print("所有任务输出:", outputs.keys())
    print("Task1 输出:", outputs['task1'].shape)
    print("Task2 输出:", outputs['task2'].shape)