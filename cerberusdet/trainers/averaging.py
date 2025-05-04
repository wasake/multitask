# 导入必要的库
import logging  # 提供日志记录功能，用于调试和记录程序运行信息
import os  # 提供与操作系统交互的功能，例如文件路径操作和环境变量获取
import random  # 提供随机数生成功能
from typing import Union, List  # 提供类型注解，用于代码可读性和静态检查

# 导入 PyTorch 和相关库
import torch  # 深度学习框架
import torch.nn as nn  # PyTorch 的神经网络模块
import torch.optim.lr_scheduler as lr_scheduler  # 学习率调度模块

# 从自定义模块导入功能
from cerberusdet.models.cerberus import CerberusDet  # 模型 CerberusDet
from cerberusdet.trainers.base_trainer import BaseTrainer  # 基础训练类
from cerberusdet.utils.general import colorstr, one_cycle  # 通用工具函数
from cerberusdet.utils.plots import plot_lr_scheduler  # 用于绘制学习率调度图
from cerberusdet.utils.torch_utils import de_parallel, get_hyperparameter  # PyTorch 工具函数

# 第三方库导入
from loguru import logger  # 先进的日志记录库
from torch.cuda import amp  # 混合精度训练工具
from tqdm import tqdm  # 进度条显示库
import warnings  # 用于忽略某些警告信息

# 忽略将来的警告信息，以保持日志输出简洁
warnings.simplefilter(action='ignore', category=FutureWarning)

# 初始化日志记录器
LOGGER = logging.getLogger(__name__)
# 从环境变量中获取本地进程的 rank，默认为 -1（非分布式模式）
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# 定义 Averaging 类，继承自 BaseTrainer
class Averaging(BaseTrainer):
    """
    Averaging 类用于实现梯度平均的训练方法。
    """

    def __init__(self, device, model, model_manager, train_loaders, val_loaders, dataset, imgsz, gs, loss_weights=None):
        """
        初始化 Averaging 类，设置训练相关的参数和超参数。

        参数:
          device: 指定训练设备（CPU 或 GPU）。
          model: 要训练的模型。
          model_manager: 模型管理器，负责保存、加载模型等操作。
          train_loaders: 训练数据加载器列表，每个任务对应一个加载器。
          val_loaders: 验证数据加载器列表。
          dataset: 数据集对象。
          imgsz: 输入图像大小。
          gs: 归一化尺寸。
          loss_weights: 损失权重字典（可选），指定每个任务的损失贡献权重。
        """
        # 调用父类的初始化方法，传入相关参数
        super().__init__(
            device=device,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            model_manager=model_manager,
            imgsz=imgsz,
            gs=gs,
            dataset=dataset,
        )

        # 设置是否使用 GPU 进行训练
        self.cuda = device.type != "cpu"
        # 获取批次大小，可以是单个整数或多个任务的列表
        self.batch_size: Union[int, List[int]] = self.opt.batch_size
        # 判断是否启用跳过批次的功能，用于加速训练
        self.use_batch_skipping = self.opt.skip_batches

        # 设置名义批次大小，用于归一化学习率
        self.nbs = 64
        # 获取每个训练加载器的总样本数
        datasets_len = [len(train_loader) for train_loader in self.train_loaders]
        if LOCAL_RANK in [-1, 0]:  # 如果是主进程，记录日志
            logger.info(f"数据集大小: {datasets_len} ")
        # 获取数据集中的最大批次数（用于确定训练迭代次数）
        self.nb = max(datasets_len)

        # 如果启用了跳过批次功能，根据数据集大小调整任务迭代频率
        if self.use_batch_skipping:
            max_size = max(datasets_len)
            self.iters_per_task = [max_size // datasets_len[i] for i in range(len(datasets_len))]
            if LOCAL_RANK in [-1, 0]:
                logger.info(f"查看任务迭代频率: {self.iters_per_task}")

        # 设置热身迭代次数，最大值为 3 个 epoch 或 1000 次迭代
        self.nw = max(round(get_hyperparameter(self.hyp, "warmup_epochs") * self.nb), 1000)

        # 初始化优化器
        self.optimizer = get_optimizer(model, self.hyp)
        # 获取学习率调度器
        self.scheduler, self.lf = get_lr_scedule(self.optimizer, self.hyp, self.opt.linear_lr, model_manager.epochs)
        # 混合精度训练的梯度缩放器（用于提高训练效率）
        self.scaler = amp.GradScaler(enabled=self.cuda)

        # 如果未指定损失权重，默认所有任务的损失权重为 1.0
        if loss_weights is None:
            loss_weights = dict(zip(self.task_ids, [1.0] * len(self.task_ids)))
            if LOCAL_RANK in [-1, 0]:
                logger.info(f"使用任务损失权重: {loss_weights} ")

        # 将损失权重转换为张量（便于后续计算）
        self.loss_weights = dict((k, torch.tensor(v, device=device)) for k, v in loss_weights.items())
        # 从模型管理器中获取数据集类别数
        self.nc = self.model_manager.data_dict["nc"]

    def resume(self, ckpt, optimizer_state=None):
        """
        从检查点恢复训练。

        参数:
          ckpt: 检查点字典，包含模型状态、优化器状态等信息。
          optimizer_state: 外部传入的优化器状态，优先级高于ckpt中的优化器状态。

        返回:
          start_epoch: 恢复后的起始 epoch（默认为 0）。
        """
        # 调用父类的resume方法，处理优化器状态和其他训练参数
        return super().resume(ckpt, optimizer_state)

    def get_optimizer_dict(self):
        """
        获取优化器的状态字典。

        返回:
          优化器的当前状态，用于保存和恢复训练过程。
        """
        return self.optimizer.state_dict()

    def train_epoch(self, model, ema, epoch, local_rank, world_size):
        """
        训练一个 epoch。

        参数:
          model: 当前正在训练的模型。
          ema: 指数移动平均模型，用于平滑模型权重。
          epoch: 当前的训练 epoch 编号。
          local_rank: 当前进程的 rank（分布式训练中用于标识进程）。
          world_size: 总进程数（分布式训练使用）。
        """
        # 如果模型是 CerberusNet，基于当前 epoch 冻结或解冻共享层
        if hasattr(de_parallel(model), "heads"):  # 检查模型是否有 "heads" 属性
            if epoch < self.opt.freeze_shared_till_epoch:
                CerberusDet.freeze_shared_layers(model)  # 冻结共享层
            elif 0 < self.opt.freeze_shared_till_epoch == epoch:
                CerberusDet.unfreeze_shared_layers(model)  # 解冻共享层

        # 如果不是演化模式，则启用绘图功能
        plots = not self.opt.evolve  # 如果不在演化模式下，则启用绘图功能
        model.train()  # 设置模型为训练模式

        loader_iterators = []  # 初始化数据加载器迭代器列表
        # 遍历所有任务，初始化数据加载器的迭代器
        for task_i, task in enumerate(self.task_ids):
            data_loader = self.train_loaders[task_i]  # 获取当前任务的数据加载器
            if local_rank != -1:
                # 如果是分布式训练，设置每个 epoch 的数据加载器种子
                data_loader.sampler.set_epoch(epoch)  # 设置数据加载器的 epoch 以确保不同进程数据不冲突
            loader_iterators.append(iter(data_loader))  # 将数据加载器迭代器添加到列表

        # 重置打印信息
        self.reset_print_info(local_rank)  # 重置用于打印的训练信息
        pbar = enumerate(range(self.nb))  # 枚举训练迭代次数，`self.nb` 是当前 epoch 中的批次数

        # 初始化每个控制块的分支数
        num_branches = dict()  # 创建一个字典来存储每个控制块的分支数
        for idx, (ctrl, block) in enumerate(de_parallel(model).control_blocks()):
            n_branches = max(len(ctrl.serving_tasks), 1.0)  # 每个控制块至少有一个分支
            num_branches[idx] = torch.tensor(n_branches, device=self.device)  # 将分支数作为张量存储

        progress_bar = tqdm(total=self.nb) if local_rank in [-1, 0] else None  # 初始化进度条，用于显示训练进度

        # 初始化优化器梯度
        self.optimizer.zero_grad()  # 清空之前的梯度
        log_step = self.nb // 10 if self.opt.evolve else 1  # 如果是演化模式，每 10 次打印一次日志，否则每次打印一次

        # 遍历每个批次进行训练
        for i, batch_idx in pbar:
            ni = i + self.nb * epoch  # 当前迭代的全局索引，`i` 为当前批次，`self.nb * epoch` 用于区分不同的 epoch

            # 热身阶段调整学习率
            if ni <= self.nw:
                BaseTrainer.warmup_lr(ni, epoch, self.optimizer, self.nw, self.hyp, self.lf)  # 在热身阶段使用学习率调整函数

            printed_task_i = random.randint(0, len(self.task_ids) - 1)  # 随机选择一个任务用于日志打印

            skipped_tasks = []  # 用于存储被跳过的任务
            # 遍历每个任务，计算梯度并累加
            for task_i, task_id in enumerate(self.task_ids):

                if self.use_batch_skipping and i % self.iters_per_task[task_i] != 0:
                    skipped_tasks.append(task_id)  # 如果启用跳过批次功能，跳过该任务
                    continue  # 跳过当前任务

                try:
                    batch = next(loader_iterators[task_i])  # 获取下一个数据批次
                except StopIteration:
                    # 如果数据加载器耗尽，重新初始化迭代器
                    loader_iterators[task_i] = iter(self.train_loaders[task_i])
                    batch = next(loader_iterators[task_i])  # 重新获取下一个数据批次

                batch = self.preprocess_batch(batch)  # 数据预处理

                # 前向传播
                with amp.autocast(enabled=self.cuda):  # 启用混合精度计算，减少内存消耗并加速计算
                    output = model(batch["img"], task_id)  # 通过模型计算输出

                    # 计算损失
                    loss, loss_items = self.compute_loss(output, batch, task_id)  # 计算当前批次的损失
                    if local_rank != -1:
                        loss *= world_size  # 在分布式训练中对损失进行缩放

                    wloss = self.loss_weights[task_id] * loss  # 根据任务的权重加权损失half

                # 反向传播
                self.scaler.scale(wloss).backward()  # 使用混合精度的梯度缩放器进行反向传播

                n_lbls = batch["bboxes"].shape[0]  # 获取当前批次的目标数量
                self._log_info(epoch, task_id, local_rank, loss_items, n_lbls, batch["img"].shape[-1])  # 记录训练日志

                # 更新进度条
                if local_rank in [-1, 0]:
                    if printed_task_i == task_i and (i % log_step == 0 and i > 0 or i == self.nb - 1):
                        progress_bar.update(log_step)  # 更新进度条
                        progress_bar.set_description(self.stat[task_id])  # 设置进度条的描述信息

                    # 如果启用绘图功能，绘制训练图像
                    if plots:
                        self.model_manager.plot_train_images(i, task_id, batch, de_parallel(model))

            # 如果启用跳过批次功能，更新控制块的分支数
            if self.use_batch_skipping:
                num_branches = dict()  # 创建一个字典来存储每个控制块的分支数
                for idx, (ctrl, block) in enumerate(de_parallel(model).control_blocks()):
                    serving_tasks = list(ctrl.serving_tasks.keys())  # 获取当前控制块正在服务的任务
                    for t in skipped_tasks:
                        if t in serving_tasks:
                            serving_tasks.remove(t)  # 从分支中移除被跳过的任务
                    n_branches = max(len(serving_tasks), 1.0)  # 更新分支数
                    num_branches[idx] = torch.tensor(n_branches, device=self.device)  # 将分支数作为张量存储

            # 执行优化器更新步骤
            self.optimizer_step(model, ema, num_branches)  # 更新优化器参数

        # 调整学习率
        lr = [x["lr"] for x in self.optimizer.param_groups]  # 获取当前学习率
        self.scheduler.step()  # 调用学习率调度器，调整学习率

        # 主进程记录日志
        if local_rank in [-1, 0]:
            for task_i, task in enumerate(self.task_ids):
                self.model_manager.train_log(task, lr, self.mloss_per_task[task_i], epoch,
                                             self.stat[task])  # 记录每个任务的训练日志

    def optimizer_step(self, model, ema, num_branches):
        """
        执行优化器更新步骤，包括梯度裁剪和梯度平均。

        参数:
          model: 当前模型。
          ema: 指数移动平均模型。
          num_branches: 每个控制块的分支数量。
        """
        self.scaler.unscale_(self.optimizer)  # 取消梯度缩放
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # 对梯度进行裁剪，防止梯度爆炸

        # 平均化控制块中的梯度
        for idx, (_, block) in enumerate(de_parallel(model).control_blocks()):
            for p_name, p in block.named_parameters():
                if not p.requires_grad or p.grad is None:
                    continue
                p.grad /= num_branches[idx]  # 将梯度除以分支数量

        self.scaler.step(self.optimizer)  # 执行优化器更新
        self.scaler.update()  # 更新梯度缩放器
        self.optimizer.zero_grad()  # 清空优化器的梯度
        if ema:
            ema.update(model)  # 更新指数移动平均模型

def init_optimizer(g, lr, momentum, name="SGD"):
    """
    初始化优化器，根据指定的名称创建优化器实例。

    参数:
      g: 参数组列表，包含需要优化的参数。
      lr: 学习率。
      momentum: 动量参数，用于加速梯度下降。
      name: 优化器的名称，默认使用 SGD（随机梯度下降）。

    返回:
      optimizer: 初始化后的优化器对象。
    """
    if name == "Adam":
        # 使用 Adam 优化器，betas 参数控制一阶和二阶动量
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))
    elif name == "AdamW":
        # 使用 AdamW 优化器，与 Adam 相似但支持权重衰减（正则化）
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == "RMSProp":
        # 使用 RMSProp 优化器，适合处理非平稳目标的优化问题
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == "SGD":
        # 使用 SGD 优化器，支持 Nesterov 动量
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        # 如果传入的优化器名称未实现，则抛出异常
        raise NotImplementedError(f"优化器 {name} 未实现。")

    return optimizer

def get_optimizer(model, hyp, name="SGD"):
    """
    根据模型和超参数配置获取优化器。

    参数:
      model: 要优化的模型。
      hyp: 超参数字典，包含学习率、动量等设置。
      name: 优化器名称，默认为 SGD。

    返回:
      optimizer: 初始化的优化器对象。
    """
    # 从超参数中获取权重衰减、学习率和动量
    decay = get_hyperparameter(hyp, "weight_decay")  # 权重衰减
    lr = get_hyperparameter(hyp, "lr0")  # 初始学习率
    momentum = get_hyperparameter(hyp, "momentum")  # 动量参数
# 参数分组：g[0] 使用衰减的权重；g[1] 不使用衰减的正则化层权重；g[2] 偏置
    g = [], [], []
    # 获取所有正则化层（例如 BatchNorm2d）
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    # 遍历模型中的所有模块，将参数分类到不同的组
    for _, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # 偏置
            g[2].append(v.bias)
        if isinstance(v, bn):  # 正则化层权重（不使用衰减）
            g[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # 普通权重
            g[0].append(v.weight)

    # 初始化优化器
    optimizer = init_optimizer(g, lr, momentum, name)

    # 为参数组添加权重衰减
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # 使用衰减
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # 不使用衰减

    # 打印优化器的信息
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}) with parameter groups "
        f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias"
    )

    return optimizer


def get_lr_scedule(optimizer, hyp, use_linear_lr, epochs):
    """
    创建学习率调度器，用于动态调整学习率。

    参数:
      optimizer: 优化器实例。
      hyp: 超参数字典，包含学习率相关的设置。
      use_linear_lr: 是否使用线性学习率衰减。
      epochs: 总的训练轮数。

    返回:
      scheduler: 学习率调度器。
      lf: 学习率函数，用于定义学习率的变化方式。
    """
    lrf = get_hyperparameter(hyp, "lrf")  # 最终的学习率倍率
    if use_linear_lr:
        # 定义线性学习率衰减函数：从初始值线性降低到最终倍率
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf
    else:
        # 使用余弦退火策略调整学习率
        lf = one_cycle(1, lrf, epochs)  # 从 1 -> lrf

    # 使用学习率函数创建调度器
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # 绘制学习率变化图，用于可视化学习率计划
    plot_lr_scheduler(optimizer, scheduler, epochs)

    return scheduler, lf