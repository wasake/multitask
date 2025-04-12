import itertools
import logging
import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from cerberusdet.models.common import Bottleneck  # noqa: F401
from cerberusdet.models.common import (  # noqa: F401
    C2,
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    BottleneckCSP,
    C2f,
    Concat,
    Contract,
    Conv,
    DWConv,
    Expand,
    Focus,
)
from cerberusdet.models.yolo import Detect, Model, get_next_layer_from_cfg, initialize_weights
from cerberusdet.utils.torch_utils import de_parallel, fuse_conv_and_bn
from loguru import logger

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
RANK = int(os.getenv("RANK", -1))

class Controller:
    """
    CerberusDet中的块控制器。保存有关其在块列表中的索引、执行链（在此块之前应该执行的块）和该块的子块的信息。

    属性:
      index:             当前块在CerberusDet.blocks中的索引
      execution_chain:   需要在当前块之前执行的块的索引列表
      parent_index:      当前块的父块索引（在CerberusDet.blocks中的位置）
      children_indices:  当前块的子块索引列表
      task_id:           如果当前块是一个头块，保存task_id
      serving_tasks:     一个字典，格式为{task_id: 某些任务的其他信息}
    """

    def __init__(self, index=None):
        """
        初始化Controller对象

        参数:
            index (int, optional): 当前控制器的索引，默认为None
        """
        self.index = index  # 当前控制器的索引
        self.execution_chain = [index]  # 执行链，初始化时只有自身
        self.parent_index = None  # 父块的索引，初始化为None
        self.children_indices = []  # 子块的索引列表，初始化为空列表
        self.task_id = None  # 当前块的任务ID，初始化为None
        self.serving_tasks = dict()  # 服务的任务字典，初始化为空字典

    def stack_on(self, controller):
        """
        将当前控制器堆叠到另一个控制器之上

        参数:
            controller (Controller): 需要堆叠到当前控制器上的控制器

        返回:
            Controller: 当前控制器，更新了执行链和父块
        """
        prev_chain = controller.execution_chain.copy()  # 获取当前控制器的执行链副本
        self.execution_chain = prev_chain + [self.index]  # 当前执行链为父控制器执行链 + 当前块的索引
        self.parent_index = controller.index  # 当前控制器的父块索引为传入控制器的索引
        controller.children_indices.append(self.index)  # 将当前块的索引添加到父控制器的子块列表
        return self

    def add_parent(self, controller, controllers):
        """
        为当前控制器扩展父块

        参数:
            controller (Controller): 需要成为当前控制器父块的控制器
            controllers (list): 所有控制器的列表
        """
        # 如果当前控制器已经有此父块，直接返回
        if self.parent_index == controller.index:
            return
        if isinstance(self.parent_index, list) and controller.index in self.parent_index:
            return
        if self.parent_index is None:
            return self.stack_on(controller)  # 如果当前没有父块，堆叠到此控制器之上

        # 将父块索引扩展到列表中
        new_chain = controller.execution_chain.copy()  # 获取父控制器的执行链副本
        if isinstance(self.parent_index, int):
            self.parent_index = [self.parent_index, controller.index]
        else:
            self.parent_index = [*self.parent_index, controller.index]  # 将父块索引添加到列表

        # 如果当前控制器的索引没有出现在父控制器的子块中，添加进去
        if self.index not in controller.children_indices:
            controller.children_indices.append(self.index)
        new_chain.append(self.index)  # 将当前控制器的索引添加到新的执行链中

        # 合并两个执行链，保证执行顺序正确
        n_elemnts = len(np.unique(new_chain + self.execution_chain))  # 合并后的元素数量
        merged_chain = []  # 最终的合并执行链
        index_left = index_right = 0
        while len(merged_chain) < n_elemnts:  # 合并两个执行链，去掉重复元素
            new_ind = new_chain[index_left]
            old_ind = self.execution_chain[index_right]
            if old_ind == new_ind:
                index_right += 1
                index_left += 1
                merged_chain.append(old_ind)
            else:
                if old_ind in controllers[new_ind].execution_chain:
                    if old_ind not in merged_chain:
                        merged_chain.append(old_ind)
                    index_right += 1
                else:
                    if new_ind not in merged_chain:
                        merged_chain.append(new_ind)
                    index_left += 1

            if index_right == len(self.execution_chain):
                merged_chain += new_chain[index_left:]  # 将剩余的父控制器执行链添加到合并链中
                break
            elif index_left == len(new_chain):
                merged_chain += self.execution_chain[index_right:]  # 将剩余的当前控制器执行链添加到合并链中
                break

        self.execution_chain = merged_chain  # 更新当前控制器的执行链

        return self

    def __str__(self):
        """
        返回当前控制器的字符串表示
        """
        return "({}): parent={}, children={}, serving=[{}]".format(
            self.index,  # 控制器的索引
            self.parent_index,  # 父控制器索引
            self.children_indices,  # 子控制器索引列表
            ", ".join(str(task_id) for task_id in self.serving_tasks),  # 服务的任务ID
        )

    def __repr__(self):
        """
        返回当前控制器的字符串表示，用于调试和输出
        """
        return str(self)

    def serialize(self):
        """
        将当前控制器对象序列化为普通的Python字典对象
        """
        return self.__dict__

    def deserialize(self, serialized_controller):
        """
        从Python字典对象反序列化当前控制器

        参数:
            serialized_controller (dict): 被序列化的控制器字典对象

        返回:
            Controller: 当前控制器对象
        """
        for k, v in serialized_controller.items():
            setattr(self, k, v)  # 将字典中的键值对赋值给当前控制器的属性
        return self


class CerberusDet(nn.Module):
    """
    CerberusDet的基础配置类，构建和组织神经网络的各个部分，包括backbone、neck和head。

    配置结构如下：
      (neck_n+1) (..) (N)  <-  heads
       |           |   |
       +---+---|---+---+
                   |
                (neck_n)
                   |
                  ...
                   |
                  (1)            <-  neck
                   |
                  (0)            <-  backbone
                   |
                  (*)            <-  input

    主要组件包括：
      - backbone: 网络的主干部分
      - neck: 连接backbone和head的部分，通常用于特征融合
      - heads: 任务相关的输出层，如检测、分类等
    """

    def __init__(self, task_ids, nc, cfg="yolov5s.yaml", ch=3, verbose=True):
        """
        初始化CerberusDet对象。

        参数:
            task_ids (list): 任务ID列表，表示不同的任务头
            nc (int): 类别数量
            cfg (str, optional): 配置文件路径，默认为"yolov5s.yaml"
            ch (int, optional): 输入通道数，默认为3（RGB图像）
            verbose (bool, optional): 是否启用详细打印，默认为True
        """
        super().__init__()
        self.blocks = nn.ModuleList()  # 存储网络的各个块
        self.controllers = list()  # 存储控制器，用于管理块之间的依赖关系
        self.heads = dict()  # 存储任务的head
        self.rep_tensors = dict()  # 存储中间结果
        self.branching_points = set()  # 分支点
        self.verbose = verbose  # 是否启用详细打印

        # 初始化backbone（骨干网络）
        backbone = Model(cfg=cfg, ch=ch, nc=nc, without_head=True, verbose=self.verbose)
        model = self.add_block(backbone)

        # 获取backbone配置的深度和宽度
        self.gd, self.gw = backbone.yaml["depth_multiple"], backbone.yaml["width_multiple"]
        self.max_channels = backbone.yaml.get("max_channels", 1024)
        nc = backbone.yaml["nc"]  # 任务的类别数量

        self.neck_head_save = []  # 保存一些块的输入
        # 解析neck部分（连接backbone和head的部分）
        model, layer_ind_map = self.parse_neck(model, deepcopy(backbone.yaml), backbone.saved_ch, nc)
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print("Finish neck parsing")

        # 解析heads部分（任务相关的输出层）
        self.parse_heads(
            model, deepcopy(backbone.yaml), task_ids, backbone.saved_ch, backbone.inplace, layer_ind_map, deepcopy(nc)
        )
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print("Finish heads parsing")

        del backbone.saved_ch
        # 对所有的模块进行权重初始化
        for block in self.blocks:
            initialize_weights(block)

        self.yaml = backbone.yaml  # 保存backbone的配置
        self.build()  # 构建整个网络结构
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print(self.info())  # 打印网络信息

    def test_forward(self, device=None):
        """
        测试前向传播。用于检查网络是否正常工作，不用于训练时（避免在EMA创建期间出错）。

        参数:
            device (torch.device, optional): 设备，如果为None则使用当前设备
        """
        s = 256  # 输入尺寸
        test_input = torch.ones(1, 3, s, s)  # 创建一个全1的输入张量
        if device is not None:
            test_input = test_input.to(device)  # 将输入移动到指定设备
        self.forward(test_input)  # 执行前向传播

    def parse_neck(self, prev_model, cfg, ch, nc):
        """
        解析网络的neck部分，构建从backbone到head的中间层。

        参数:
            prev_model (Module): 前一个模型（backbone的输出）
            cfg (dict): 配置字典
            ch (list): 通道数列表
            nc (int): 类别数量

        返回:
            prev_model (Module): 解析后的模型
            layer_ind_map (dict): 图层索引映射
        """
        ind = len(cfg["backbone"])  # 获取backbone部分的最后一个索引
        layer_ind_map = {}  # 存储图层索引映射

        for i, layer in enumerate(cfg["neck"], start=1):
            f, n, m, args = layer  # 从、数字、模块、参数
            m = eval(m) if isinstance(m, str) else m  # 如果是字符串，eval转为模块
            args, _, n, c2, m_ = get_next_layer_from_cfg(self.gd, ch, self.gw, nc, m, n, f, args, self.max_channels)

            ch.append(c2)  # 更新通道数

            t = str(m)[8:-2].replace("__main__.", "")  # 获取模块类型
            np = sum([x.numel() for x in m_.parameters()])  # 获取模块的参数数量

            # 创建下一个块，并将其堆叠到前一个模型
            next_block = self.add_block(m_).stack_on(prev_model)
            f = [f] if isinstance(f, int) else f  # 如果是整数则转为列表
            assert f[0] == -1 or len(f) == 1, "Unsupported config"

            new_input_idx = []  # 新的输入索引列表
            for x in f:
                if x != -1 and x >= len(cfg["backbone"]):
                    x = layer_ind_map[x]  # 如果是中间层，使用层的索引映射
                    self.neck_head_save.append(x)
                    next_block.add_parent(self.controllers[x], self.controllers)
                elif x == -1 and i == 1:
                    raise ValueError("Input for first cerbernet block must be defined")
                elif x != -1 and x < len(cfg["backbone"]):
                    x = (0, x)  # 如果是backbone中的层，使用（0，x）作为输入索引
                    next_block.add_parent(self.controllers[0], self.controllers)
                new_input_idx.append(x)

            m_.i, m_.f, m_.type, m_.np = ind, new_input_idx, t, np  # 记录原始索引、输入索引、模块类型、参数数量

            if self.verbose and LOCAL_RANK in [-1, 0]:
                LOGGER.info("%3s-%3s%18s%3s%10.0f  %-40s%-30s" % (i, ind, f, n, np, t, args))  # 打印模块信息

            prev_model = next_block  # 更新前一个模型
            layer_ind_map[ind] = next_block.index  # 更新图层索引映射

            ind += 1

        return prev_model, layer_ind_map

    def parse_heads(self, prev_model, cfg, task_ids, ch, inplace, layer_ind_map, nc):
        """
        解析网络的head部分，为每个任务添加对应的输出层。

        参数:
            prev_model (Module): 前一个模型（neck部分的输出）
            cfg (dict): 配置字典
            task_ids (list): 任务ID列表
            ch (list): 通道数列表
            inplace (bool): 是否使用原地操作
            layer_ind_map (dict): 图层索引映射
            nc (int): 类别数量
        """
        ind = len(cfg["backbone"]) + len(cfg["neck"])  # 获取backbone和neck部分的总长度
        if len(cfg["head"]) != 1:
            raise NotImplementedError  # 当前只支持单个head

        for task_id in task_ids:
            for i, layer in enumerate(cfg["head"]):
                f, n, m, args = layer  # 从、数字、模块、参数
                m = eval(m) if isinstance(m, str) else m  # 如果是字符串，eval转为模块

                if self.verbose and LOCAL_RANK in [-1, 0]:
                    print(f"Adding head for {task_id}")
                args_, nc, n, _, m_ = get_next_layer_from_cfg(
                    self.gd, ch, self.gw, nc, m, n, f, deepcopy(args), self.max_channels
                )
                h = self.add_head(m_, task_id)  # 添加head

                t = str(m)[8:-2].replace("__main__.", "")  # 获取模块类型
                np = sum([x.numel() for x in m_.parameters()])  # 获取模块的参数数量
                f = [f] if isinstance(f, int) else f  # 如果是整数则转为列表

                if f[0] == -1:
                    h.stack_on(prev_model)  # 如果输入为-1，将head堆叠到前一个模型

                new_input_idx = []  # 新的输入索引列表
                for x in f:
                    if x != -1 and x >= len(cfg["backbone"]):
                        x = layer_ind_map[x]  # 如果是中间层，使用层的索引映射
                        self.neck_head_save.append(x)
                        h.add_parent(self.controllers[x], self.controllers)
                    elif x != -1 and x < len(cfg["backbone"]):
                        raise ValueError("Input for the head must be from neck")
                    new_input_idx.append(x)

                m_.i, m_.f, m_.type, m_.np = (
                    ind,
                    new_input_idx,
                    t,
                    np,
                )  # 记录原始索引、输入索引、模块类型、参数数量

                if m in [Detect]:
                    s = 256  # 设置最小步幅为256
                    m_.inplace = inplace  # 是否使用原地操作
                    m_.stride = torch.tensor(
                        [s / x.shape[-2] for x in self.forward(torch.zeros(1, cfg["ch"], s, s), task_id)]
                    )  # 设置步幅

                    if not hasattr(self, "stride"):
                        self.stride = m_.stride  # 初始化步幅
                    else:
                        assert torch.equal(m_.stride, self.stride)

                    m_.bias_init()  # 初始化bias，只运行一次

                if self.verbose and LOCAL_RANK in [-1, 0]:
                    LOGGER.info("%3s-%3s%18s%3s%10.0f  %-40s%-30s" % (i, ind, f, n, np, t, args_))  # 打印模块信息
    def set_task(self, task_id):
        """
        设置当前任务的ID。用于在模型的前向传播过程中确定要执行的任务。
        DO NOT USE: 请不要将task_id作为参数传递到forward方法中，而是通过调用此方法来设置任务ID。

        参数:
          task_id: 当前任务的ID
        """
        self.cur_task = task_id  # 设置当前任务ID

    def get_head(self, task_id) -> Detect:
        """
        获取指定任务的输出头模块。

        参数:
          task_id: 任务的ID

        返回:
          Detect: 任务对应的检测头模块
        """
        indx = self.heads[task_id]  # 获取任务ID对应的输出头模块的索引
        return self.blocks[indx]  # 返回对应的模块

    def add_block(self, module):
        """
        注册一个新的CerberusDet模块，并将其自动添加到模型的块（blocks）和执行图中。

        参数:
          module: 一个 `nn.Module` 对象，表示新加入的模块

        返回:
          Controller: 返回新添加模块的控制器对象
        """
        new_index = len(self.blocks)  # 新模块的索引是当前模块数量
        new_controller = Controller(new_index)  # 创建一个新的控制器
        self.blocks.append(module)  # 将模块添加到模型的模块列表中
        self.controllers.append(new_controller)  # 将控制器添加到控制器列表中
        return new_controller  # 返回新的控制器

    def add_head(self, module, task_id):
        """
        将一个新的CerberusDet模块作为输出头（Head）注册。与 `add_block()` 方法类似，
        但会将控制器添加到 `self.heads` 中。

        参数:
          module: 一个 `nn.Module` 对象，表示新加入的头部模块
          task_id: 任务的ID，表示该头部解决的任务

        返回:
          Controller: 返回新添加模块的控制器对象
        """
        new_controller = self.add_block(module)  # 调用add_block方法添加模块
        new_controller.task_id = task_id  # 将任务ID赋给控制器
        self.heads[task_id] = new_controller.index  # 将头部任务ID与模块索引关联
        return new_controller  # 返回新的控制器

    def info(self):
        """
        获取模型的基本信息，返回模型的控制器信息和头部信息。

        返回:
          str: 返回包含模型控制器和头部信息的字符串
        """
        # 获取控制器的信息
        items = "\n  ".join(str(c) for c in self.controllers)
        controllers = "(block controllers):\n  " + items
        # 获取头部的信息
        items = "\n  ".join("({}) -> {}  {}".format(k, str(c), type(self.blocks[c])) for k, c in self.heads.items())
        heads = "(heads):\n  " + items
        return controllers + "\n" + heads  # 返回控制器和头部信息的拼接字符串

    def execution_plan(self, task_ids: Union[List[str], str]):
        """
        动态地构建执行计划，根据任务ID确定执行顺序。

        参数:
          task_ids: 任务ID或任务ID列表，指定需要执行的任务

        返回:
          execution_order: 需要执行的模块索引列表
          branching_ids: 分支点的索引集合
        """
        if not isinstance(task_ids, list):
            task_ids = [task_ids]  # 确保task_ids是一个列表
        execution_order = []  # 用于存储执行顺序
        branching_ids = set()  # 存储分支点的索引集合
        for task_id in task_ids:
            branching_point = None
            controller = self.controllers[self.heads[task_id]]  # 获取对应任务的控制器
            task_exec_chain = controller.execution_chain  # 获取该任务的执行链
            for i, index in enumerate(task_exec_chain):
                if index not in execution_order:
                    break
                branching_point = index
            execution_order += task_exec_chain[i:].copy()  # 将执行链中的模块加入执行顺序
            if branching_point is not None:
                parents = self.controllers[index].parent_index  # 获取父节点
                if isinstance(parents, int):
                    assert parents == branching_point
                    branching_ids.add(branching_point)
                else:
                    branching_ids.update(parents)
        return execution_order, branching_ids  # 返回执行顺序和分支点索引

    def control_blocks(self, task_ids=None):
        """
        迭代器，遍历模型中的模块。如果指定了 `task_ids`，则仅遍历与指定任务相关的模块。

        参数:
          task_ids: 任务ID列表，指定需要遍历的任务

        返回:
          迭代器：返回控制器和模块的迭代器
        """
        if task_ids is None:
            for controller, block in zip(self.controllers, self.blocks):
                yield controller, block  # 遍历所有模块
        else:
            execution_order, _ = self.execution_plan(task_ids)  # 获取执行顺序
            for index in execution_order:
                yield self.controllers[index], self.blocks[index]  # 按照执行顺序遍历模块

    def parameters(self, recurse=True, task_ids=None, only_trainable=False):
        """
        返回模型的参数迭代器。如果指定了 `task_ids`，则仅返回与任务输出相关的参数。

        参数:
          recurse: 是否递归遍历子模块的参数
          task_ids: 是否仅返回与指定任务相关的参数
          only_trainable: 是否仅返回可训练的参数

        返回:
          迭代器：返回模块参数的迭代器
        """
        if task_ids is None and not only_trainable:
            for param in super().parameters(recurse):
                yield param  # 返回所有参数
        else:
            if task_ids is None:
                task_ids = list(self.heads.keys())  # 默认遍历所有任务的头部
            execution_order, _ = self.execution_plan(task_ids)  # 获取执行顺序
            for index in execution_order:
                if only_trainable:
                    if not hasattr(self.blocks[index], "trainable"):
                        continue
                    if self.blocks[index].trainable is not True:
                        continue

                for param in self.blocks[index].parameters():
                    yield param  # 返回参数

    def build(self):
        """
        构建模型，设置每个控制器的任务。
        """
        for _, head_index in self.heads.items():
            controller = self.controllers[head_index]  # 获取控制器
            task_id = controller.task_id  # 获取任务ID
            for index in controller.execution_chain:  # 遍历执行链中的所有模块
                idx = len(self.controllers[index].serving_tasks)  # 获取任务索引
                self.controllers[index].serving_tasks[task_id] = idx  # 设置任务在模块中的索引
        _, self.branching_points = self.execution_plan(list(self.heads.keys()))  # 获取分支点的索引

    def create_nested_branch(
            self,
            index: int,
            branches: List[int],
            device: Optional[torch.device] = None,
            inds_to_map_per_head: Optional[Dict[int, List[int]]] = None,
            next_ids_map: Optional[Dict[int, Dict[int, int]]] = None
    ):
        """
        动态克隆指定索引的模块的子模块，并根据提供的 `branches` 列表堆叠克隆的子模块，
        形成新的分支。

        [操作前]                           [操作后]
                    __ ...........           --1----2- ...........
        index      /                        / index
        --O---1---2--- branches[0]       --O            __ branches[0]
                   \__                      \ 克隆     /
                       branches[1]           --1----2--- branches[1]

        参数:
          index:      要克隆的模块的索引
          branches:   需要堆叠到新克隆模块上的分支模块的索引
          device:     克隆模块要放置的设备，如果未指定可以稍后决定
          inds_to_map_per_head: 每个头（任务）对应的模块索引映射，需要映射到新的模块索引
                               例如: {branches[0]: [2]} 表示在 `branches[0]` 头上，模块索引 2 需要被映射到新索引
          next_ids_map: 用于保存每个头对应的模块索引映射关系
                       例如: {branches[0]: {2: 4}} 表示对于 `branches[0]` 头，模块索引 2 被映射为索引 4

        返回:
          controllers: 新分支模块的控制器
          blocks:      新分支模块的结构
        """
        # 如果index已经是一个头模块，抛出异常
        if index in self.heads:
            raise ValueError("Cannot split 's head.")

        if inds_to_map_per_head is not None:
            assert next_ids_map is not None

        start_controller = self.controllers[index]  # 获取起始模块的控制器

        branches_names = [task_id for task_id, task_ind in self.heads.items() if task_ind in branches]
        if len(branches_names) != len(branches):
            raise ValueError("Indices of branches must be indexes of heads.")

        cloned_blocks = []  # 克隆模块列表
        cloned_controllers = []  # 克隆控制器列表

        # 获取任务的执行顺序
        exec_order, _ = self.execution_plan(branches_names)
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print(f"\nOld exec plan for {branches_names} : {exec_order}")

        clones_ids = {}  # 用于保存克隆模块的映射关系

        prev_model = start_controller  # 上一个模型
        prev_controller = start_controller  # 上一个控制器
        for ind in exec_order:
            if ind <= index:
                continue
            if ind in branches:
                break
            # 克隆模块并根据需要移动到指定设备
            cloned_block = deepcopy(self.blocks[ind])
            if device is not None:
                cloned_block = cloned_block.to(device)

            controller = self.controllers[ind]
            new_index = len(self.controllers)  # 计算新索引
            cloned_controller = Controller(new_index)  # 创建新控制器
            clones_ids[controller.index] = new_index  # 保存克隆模块的索引映射关系
            self.controllers.append(cloned_controller)
            self.blocks.append(cloned_block)

            # 修改父子关系
            if isinstance(controller.parent_index, int):
                cloned_controller.stack_on(prev_model)  # 将父模块堆叠到新模块上
            elif isinstance(controller.parent_index, list):
                cloned_controller.stack_on(prev_model)
                for parent_ind in controller.parent_index:
                    if parent_ind == prev_controller.index:
                        continue
                    if parent_ind in clones_ids:
                        new_parent_ind = clones_ids[parent_ind]
                        cloned_controller.add_parent(self.controllers[new_parent_ind], self.controllers)
                        if parent_ind in self.neck_head_save:
                            self.neck_head_save.append(new_parent_ind)
                    else:
                        cloned_controller.add_parent(self.controllers[parent_ind], self.controllers)

            else:
                raise ValueError("Unknown parent type")

            # 更新克隆模块的父子关系
            for i, from_ind in enumerate(cloned_block.f[:]):
                if from_ind != -1 and from_ind in clones_ids:
                    cloned_block.f[i] = clones_ids[from_ind]

            prev_model = cloned_controller  # 更新上一个模块为克隆后的控制器
            prev_controller = controller  # 更新上一个控制器

            cloned_blocks.append(cloned_block)  # 将克隆的模块加入列表
            cloned_controllers.append(cloned_controller)  # 将克隆的控制器加入列表

        # 将头模块堆叠在克隆的分支模块之上
        for head_ind in branches:
            head_controller = self.controllers[head_ind]
            head_controller.execution_chain = [head_controller.index]  # 设置执行链

            # 更新头模块的父子关系
            for i, from_ind in enumerate(self.blocks[head_ind].f[:]):
                if from_ind != -1 and from_ind in clones_ids:
                    self.blocks[head_ind].f[i] = clones_ids[from_ind]

            if isinstance(head_controller.parent_index, int):
                parent_ind = head_controller.parent_index
                assert parent_ind in clones_ids
                self.controllers[parent_ind].children_indices.remove(head_ind)  # 移除原父模块的孩子模块
                new_parent_ind = clones_ids[parent_ind]  # 获取新的父模块索引
                head_controller.stack_on(self.controllers[new_parent_ind])  # 将头模块堆叠到新父模块上

                if parent_ind in self.neck_head_save:
                    self.neck_head_save.append(new_parent_ind)  # 将新父模块添加到保存的列表中
                continue

            old_parent_inds = head_controller.parent_index
            head_controller.parent_index = None
            for i, parent_ind in enumerate(old_parent_inds):
                old_parent = self.controllers[parent_ind]
                if parent_ind in clones_ids:
                    old_parent.children_indices.remove(head_ind)  # 移除原父模块的孩子模块
                    new_parent_ind = clones_ids[parent_ind]  # 获取新的父模块索引
                    head_controller.add_parent(self.controllers[new_parent_ind], self.controllers)
                    if parent_ind in self.neck_head_save:
                        self.neck_head_save.append(new_parent_ind)
                elif head_controller.parent_index is None:
                    prev_chain = old_parent.execution_chain.copy()
                    head_controller.execution_chain = prev_chain + [head_controller.index]
                    head_controller.parent_index = old_parent.index
                    assert head_controller.index in old_parent.children_indices
                else:
                    assert head_controller.index in old_parent.children_indices
                    head_controller.add_parent(old_parent, self.controllers)

        # 更新任务服务和分支索引
        for controller in self.controllers:
            controller.serving_tasks = dict()  # 清空所有控制器的服务任务
        self.rep_tensors.clear()  # 清空缓存
        self.build()  # 重新构建模型

        exec_order, _ = self.execution_plan(branches_names)
        if self.verbose and LOCAL_RANK in [-1, 0]:
            print(
                f"\nNew exec plan for {branches_names}({branches}) : "
                f"{exec_order}\n Branching ids: {self.branching_points}"
            )

        # 更新下一个配置的索引映射
        for old_ind, new_ind in clones_ids.items():
            for task_ind in branches:
                if (
                        inds_to_map_per_head is not None
                        and task_ind in inds_to_map_per_head
                        and old_ind in inds_to_map_per_head[task_ind]
                ):
                    next_ids_map[task_ind][old_ind] = new_ind

        return cloned_controllers, cloned_blocks

    def split(self, index, branching_scheme, device, next_cerber_configs):
        """
        按照 `branching_scheme` 提供的分支方案，拆分一个模块的块为多个模块。
        例如，调用 `split(0, [[2], [3, 4], [5, 6]])` 会产生如下结构：

        | B |  (2) (3) (4) (5) (6)     | A |  (2) (3) (4) (5) (6)
        | E |   |   |   |   |   |      | F |   |   |   |   |   |
        | F |   +---+---|---+---+      | T |   |   +---|   +---|
        | O |          (1)             | E |  (1)     (6)     (7)
        | R |           |              | R |  |        |       |
        | E |          (0)             |   |  +--------|-------+
        |   |           |              |   |          (0)
        |   |          (*)             |   |           |
        |   |                          |   |          (*)

        参数:
          index:            要拆分的模块的索引
          branching_scheme: 按照指定方案拆分模块的分支配置
          device:           分支模块放置的设备

        返回:
          controllers:      拆分后各个分支模块的控制器
          blocks:           拆分后各个分支模块的结构
        """
        inds_to_map_per_head: Dict[int, List[int]] = defaultdict(list)  # 每个头的模块索引映射
        next_ids_map: Dict[int, Dict[int, int]] = {}  # 每个头的索引映射字典

        # 处理下一个配置
        for sc in next_cerber_configs:
            for head_ind in itertools.chain(*sc[1]):
                inds_to_map_per_head[head_ind].append(sc[0])  # 填充索引映射
                next_ids_map[head_ind] = {sc[0]: None}
                if head_ind in branching_scheme[0]:
                    next_ids_map[head_ind][sc[0]] = sc[0]

        controller = self.controllers[index]
        block = self.blocks[index]

        # 更新分支方案的索引
        total_branches = set()
        for branch in branching_scheme:
            total_branches.update(set(branch))

        if not total_branches == set(self.heads.values()):
            missed_inds = [ind for ind in self.heads.values() if ind not in total_branches]
            logger.warning(f"Branching config does not include {missed_inds} head inds")

        # 检查分支方案是否有重复
        for i in range(len(branching_scheme)):
            scheme_a = set(branching_scheme[i])
            for j in range(i + 1, len(branching_scheme)):
                scheme_b = set(branching_scheme[j])
                if not scheme_a.isdisjoint(scheme_b):
                    raise ValueError("The branching schemes should be disjoint to each other.")

        if self.verbose and LOCAL_RANK in [-1, 0]:
            logger.info(f"Branching ids: {self.branching_points}")

        new_controllers, new_blocks = [controller], [block]  # 存储新模块的控制器和结构
        for branch in branching_scheme[1:]:
            if self.verbose and LOCAL_RANK in [-1, 0]:
                logger.info(f"Creating branch for {branch} at {index}")
            tmp_ctrl, tmp_block = self.create_nested_branch(index, branch, device, inds_to_map_per_head, next_ids_map)
            new_controllers.append(tmp_ctrl)
            new_blocks.append(tmp_block)

        return new_controllers, new_blocks, next_ids_map

    def sequential_split(self, cerber_schedule, device):
        """
        按照 `cerber_schedule` 中的分支计划，逐步拆分模型块。
        """
        # 获取分支计划中的头模块索引
        schedule_head_ids = [list(itertools.chain(*cerber_conf[-1])) for cerber_conf in cerber_schedule]
        schedule_head_ids = list(itertools.chain(*schedule_head_ids))
        schedule_head_ids = sorted(list(np.unique(schedule_head_ids)))
        model_head_ids = sorted(list(self.heads.values()))

        assert (
                model_head_ids == schedule_head_ids or len(schedule_head_ids) == 0
        ), f"Invalid cerberusNet config {cerber_schedule}"

        # 逐步拆分
        for i in range(len(cerber_schedule)):
            branching_scheme = cerber_schedule[i]
            next_configs = cerber_schedule[i + 1:]
            _, _, ids_map = self.split(*branching_scheme, device, next_configs)

            for ii, next_branching_scheme in enumerate(next_configs):
                mapped_ind = [
                    ids_map[head_ind][next_branching_scheme[0]]
                    for head_ind in itertools.chain(*next_branching_scheme[1])
                ]
                assert None not in mapped_ind
                assert len(np.unique(mapped_ind)) == 1
                cerber_schedule[i + 1 + ii][0] = mapped_ind[0]

        # 返回

    def fuse(self):
        """
        将模型中的Conv2d()和BatchNorm2d()层融合为一个层，减少计算开销。
        如果层已经融合，则不会重复融合。

        返回:
          self: 经过融合的模型对象
        """
        # 如果模型没有verbose属性，设置为True
        if not hasattr(self, "verbose"):
            setattr(self, "verbose", True)

        # 打印融合过程的日志（如果启用verbose并且是主进程）
        if self.verbose and LOCAL_RANK in [-1, 0]:
            LOGGER.info("Fusing layers... ")

        # 遍历所有的块（模块）
        for module in self.blocks:
            if isinstance(module, Model):
                # 如果模块是一个子模型，递归融合
                module = module.fuse()
            else:
                # 如果模块不是子模型，则遍历模块中的每个子模块
                for m in module.modules():
                    if type(m) is Conv and hasattr(m, "bn"):
                        # 如果是卷积层并且有BatchNorm层，进行融合
                        m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 融合卷积和BatchNorm层
                        delattr(m, "bn")  # 删除BatchNorm层
                        m.forward = m.fuseforward  # 更新前向传播函数为融合后的版本
                        # 打印“融合层”的日志（可选）
                        # print("Fuse layer")

        return self

    def _get_one_input(self, block_layer, parent_index, parent_out, branching_ids, rep_tensors):
        """
        获取一个父模块的输出作为输入。

        参数:
          block_layer: 当前层
          parent_index: 父模块的索引
          parent_out: 父模块的输出
          branching_ids: 分支模块的索引集合
          rep_tensors: 存储各个模块输出的字典

        返回:
          输入张量列表
        """
        if parent_index in branching_ids:
            # 如果父模块是分支的一部分，使用存储的张量
            parent_out = rep_tensors[parent_index]

        if isinstance(parent_out, list):
            # 如果父模块输出是一个列表，选择一个特定的输出
            assert len(block_layer.f) == 1 and block_layer.f[0][1] != -1
            input_idx = block_layer.f[0][1]
            parent_out = parent_out[input_idx]
            assert parent_out is not None
        return [parent_out]

    def _get_several_inputs(self, block_layer, parent_index, x, branching_ids, middle_outputs, rep_tensors,
                            neck_head_save):
        """
        获取多个父模块的输出作为输入。

        参数:
          block_layer: 当前层
          parent_index: 父模块的索引
          x: 输入张量
          branching_ids: 分支模块的索引集合
          middle_outputs: 存储中间层输出的字典
          rep_tensors: 存储各个模块输出的字典
          neck_head_save: 存储用于保存Neck头输出的索引

        返回:
          输入张量列表
        """
        next_input = []
        assert len(block_layer.f) == len(parent_index)

        for input_idx, _parent_index in zip(block_layer.f, parent_index):

            if isinstance(input_idx, tuple):
                # 输入来自于backbone中间层
                assert _parent_index == 0
                parent_out = (
                    middle_outputs[_parent_index] if _parent_index not in branching_ids else rep_tensors[_parent_index]
                )

                assert isinstance(parent_out, list)
                backbone_ind = input_idx[1]
                parent_out = parent_out[backbone_ind]
            elif isinstance(input_idx, int) and input_idx == -1:
                # 输入来自前一层的输出
                parent_out = x if _parent_index not in branching_ids else rep_tensors[_parent_index]
            elif isinstance(input_idx, int) and input_idx != -1 and input_idx in neck_head_save:
                # 输入来自Neck中间层
                assert _parent_index == input_idx
                parent_out = (
                    middle_outputs[_parent_index] if _parent_index not in branching_ids else rep_tensors[_parent_index]
                )
            else:
                raise ValueError(f"Unknown input index {input_idx}")

            next_input.append(parent_out)
        return next_input

    def forward(self, input_tensor, task_ids=None, retain_tensors=False, retain_all=False):
        """
        定义每次调用时执行的计算过程。动态且自动决定执行顺序和要执行的任务。

        参数:
          input_tensor: 所有任务共同的输入张量
          task_ids: 要执行的任务的标识符
          retain_tensors: 如果为True，则保存分支张量到rep_tensors
          retain_all: 如果为True，则保存所有张量到rep_tensors

        返回:
          一个字典，键为任务ID，值为任务的输出
        """
        if task_ids is None and hasattr(self, "cur_task"):

            # 如果没有指定task_ids且当前任务已定义，则使用当前任务
            # 在epoch开始的前三次必定触发，不适合
            # print(f"WARN: forcely inference for {self.cur_task} task")
            task_ids = self.cur_task

            
        elif task_ids is None:
            # 如果没有指定task_ids，则默认执行所有头任务
            task_ids = list(self.heads.keys())

        # 获取任务执行顺序和分支模块的ID
        exec_order, branching_ids = self.execution_plan(task_ids)

        x = input_tensor
        outputs = dict()

        middle_outputs = dict()
        for index in exec_order:
            controller = self.controllers[index]
            parent_index = controller.parent_index
            block_layer = self.blocks[index]

            if parent_index is None:
                # 如果没有父模块，则输入来自于backbone
                next_input = [x]
            elif isinstance(parent_index, int):
                # 如果只有一个父模块，则获取该父模块的输出作为输入
                parent_out = x
                next_input = self._get_one_input(block_layer, parent_index, parent_out, branching_ids, self.rep_tensors)
            else:
                # 如果有多个父模块，则获取它们的输出作为输入
                next_input = self._get_several_inputs(
                    block_layer,
                    parent_index,
                    x,
                    branching_ids,
                    middle_outputs,
                    self.rep_tensors,
                    self.neck_head_save,
                )

            if len(next_input) == 1:
                next_input = next_input[0]

            # 前向传播
            x = block_layer(next_input)

            # 保存张量
            if retain_all:
                self.rep_tensors[index] = x
            elif retain_tensors and index in self.branching_points:
                self.rep_tensors[index] = x
            elif index in branching_ids:
                self.rep_tensors[index] = x

            if index in self.neck_head_save:
                assert index not in middle_outputs
                middle_outputs[index] = x
            if index == 0 and 0 not in branching_ids:
                # 保存backbone的输出
                middle_outputs[0] = x

            if controller.task_id is not None:
                outputs[controller.task_id] = x

        # 返回指定任务的输出，如果task_ids是单个任务ID，则返回该任务的输出
        return outputs[task_ids] if isinstance(task_ids, str) else outputs

    @staticmethod
    def freeze_shared_layers(cerberus_model):
        """
        冻结共享层的参数，使它们不参与训练。

        参数:
          cerberus_model: 要冻结层的Cerberus模型
        """
        model = de_parallel(cerberus_model)

        model_tasks = len(model.heads)
        if model_tasks == 1:
            # 如果只有一个任务，跳过冻结操作
            return

        if model.verbose and LOCAL_RANK in [-1, 0]:
            logger.info("freeze layers...")

        for idx, (ctrl, block) in enumerate(model.control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.0)
            if n_branches != model_tasks:
                continue
            # 冻结参数
            for _, p in block.named_parameters():
                p.requires_grad = False

            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False  # 禁用BatchNorm的统计信息
                    m.eval()  # 设置为评估模式

    @staticmethod
    def unfreeze_shared_layers(cerberus_model):
        """
        解冻共享层的参数，使它们重新参与训练。

        参数:
          cerberus_model: 要解冻层的Cerberus模型
        """
        model = de_parallel(cerberus_model)

        model_tasks = len(model.heads)
        if model_tasks == 1:
            # 如果只有一个任务，跳过解冻操作
            return

        if model.verbose and LOCAL_RANK in [-1, 0]:
            logger.info("unfreeze layers...")

        for idx, (ctrl, block) in enumerate(model.control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.0)
            if n_branches != model_tasks:
                continue
            # 解冻参数
            for _, p in block.named_parameters():
                p.requires_grad = True

            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True  # 启用BatchNorm的统计信息
                    m.train()  # 设置为训练模式
