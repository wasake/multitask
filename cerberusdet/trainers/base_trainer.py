import logging
from collections import defaultdict

import numpy as np
import torch
from cerberusdet import val
from cerberusdet.utils.loss import Loss
from cerberusdet.utils.metrics import fitness
from cerberusdet.utils.models_manager import ModelManager
from cerberusdet.utils.torch_utils import EarlyStopping, de_parallel, get_hyperparameter

LOGGER = logging.getLogger(__name__)

class BaseTrainer:
    def __init__(
        self,
        device: torch.device,  # 设备 (如 GPU 或 CPU)
        train_loaders: list,  # 训练数据加载器
        val_loaders: list,  # 验证数据加载器
        model_manager: ModelManager,  # 模型管理器
        imgsz: int,  # 输入图像大小
        gs: int,  # 网络步长（grid size）
        dataset: list,  # 数据集
    ):
        # 初始化模型管理器、超参数、训练配置等
        self.model_manager = model_manager
        self.hyp = model_manager.hyp  # 获取超参数
        self.opt = model_manager.opt  # 获取训练选项
        self.val_loaders = val_loaders  # 验证数据加载器
        self.imgsz = imgsz  # 输入图像大小
        self.gs = gs  # 网络步长
        self.train_loaders = train_loaders  # 训练数据加载器
        self.dataset = dataset  # 数据集
        self.task_ids = model_manager.task_ids  # 从 data.yaml 中读取的任务ID列表
        self.best_fitness_per_task = dict()  # 每个任务的最佳fitness分数
        self.maps_per_task = dict()  # 每个任务的mAP分数
        for i, task in enumerate(self.task_ids):
            self.best_fitness_per_task[task] = 0.0  # 初始化最佳fitness
            nc = self.model_manager.data_dict["nc"][i]  # 获取类别数量
            self.maps_per_task[task] = np.zeros(nc)  # 初始化每个任务的mAP为0
        self.best_fitness = 0.0  # 初始化全局最佳fitness
        self.last_fitness = 0.0  # 初始化上一个epoch的fitness
        self.device = device  # 训练设备

        self.stopper = EarlyStopping(patience=self.opt.patience)  # 设置早停条件

    def set_loss(self, model):
        # 设置损失函数，使用去并行化后的模型进行损失计算
        self.compute_loss = Loss(de_parallel(model), self.task_ids)

    def train_epoch(self, model, ema, epoch, local_rank, world_size):
        """训练一个epoch"""
        raise NotImplementedError  # 需要子类实现具体的训练逻辑

    def get_optimizer_dict(self):
        """获取优化器的参数字典"""
        raise NotImplementedError  # 需要子类实现具体的优化器设置

    def resume(self, ckpt):
        """从checkpoint恢复训练"""
        raise NotImplementedError  # 需要子类实现具体的恢复逻辑

    def preprocess_batch(self, batch):
        """预处理输入批次"""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255  # 将图像数据转换为浮动类型，并归一化到[0, 1]区间
        return batch

    def reset_print_info(self, local_rank, loss_size=4):
        """重置打印日志信息"""
        assert loss_size >= 4, "Invalid log parameter"  # 确保损失日志的尺寸至少为4

        self.mloss_per_task = [torch.zeros(loss_size, device=self.device) for _ in self.task_ids]  # 每个任务的损失初始化为0
        self.stat = defaultdict(str)  # 初始化状态字典
        self.task_cnt = defaultdict(int)  # 初始化任务计数器

        if local_rank in [-1, 0]:  # 只有在主进程中进行日志打印
            log_headers = "\n"
            log_headers += ("%10s" * 3) % ("task", "epoch", "gpu_mem")  # 打印任务、epoch、GPU内存信息
            log_headers += ("%10s" * 4) % ("box", "cls", "dfl", "total")  # 打印box、cls、dfl和总损失
            if loss_size > 4:
                log_headers += ("%10s" * (loss_size - 4)) % ("loss_item",) * (loss_size - 4)  # 打印额外的损失项
            log_headers += ("%10s" * 2) % ("labels", "img_size")  # 打印标签数量和图像大小
            LOGGER.info(log_headers)  # 打印日志头部信息

    def _log_info(self, epoch, task_id, local_rank, loss_items, nb, imsz):
        """记录日志信息"""
        if local_rank in [-1, 0]:  # 只有在主进程中进行日志打印
            task_i = self.task_ids.index(task_id)  # 获取任务的索引
            n_add_losses = self.mloss_per_task[task_i].shape[0] - 4  # 计算额外损失项的数量
            self.mloss_per_task[task_i] = (self.mloss_per_task[task_i] * self.task_cnt[task_id] + loss_items) / (
                self.task_cnt[task_id] + 1
            )  # 更新每个任务的平均损失
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # 获取GPU内存使用情况
            self.stat[task_id] = ("%10s" * 3 + "%10.4g" * (6 + n_add_losses)) % (
                f"{task_id}",
                f"{epoch}/{self.model_manager.epochs - 1}",
                mem,
                *self.mloss_per_task[task_i],
                nb,
                imsz,
            )  # 格式化输出日志信息
            self.task_cnt[task_id] += 1  # 更新任务计数器

    @staticmethod
    def warmup_lr(ni, epoch, optimizer, nw, hyp, lf):
        """学习率预热"""
        xi = [0, nw]  # 学习率插值的x值
        # 为每个优化器参数组设置学习率
        for j, x in enumerate(optimizer.param_groups):
            warmup_bias_lr = get_hyperparameter(hyp, "warmup_bias_lr")  # 获取偏置的预热学习率
            x["lr"] = np.interp(ni, xi, [warmup_bias_lr if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])  # 更新学习率
            if "momentum" in x:  # 如果有动量参数
                warmup_momentum = get_hyperparameter(hyp, "warmup_momentum")  # 获取预热动量
                momentum = get_hyperparameter(hyp, "momentum")  # 获取最终动量
                x["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])  # 更新动量

    def val_epoch(self, model, ema, epoch, world_size):
        """验证一个epoch"""
        if not self.opt.evolve:
            final_epoch = (epoch + 1 == self.model_manager.epochs) or self.stopper.possible_stop  # 判断是否是最后一个epoch
        else:
            final_epoch = epoch + 1 == self.model_manager.epochs  # 在进化模式下判断是否是最后一个epoch

        if not (not self.opt.noval or final_epoch):
            return {}

        plots = not self.opt.evolve  # 是否绘制图像

        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])  # 更新EMA模型属性
        fitness_per_task = dict()  # 每个任务的fitness分数
        results_per_task = dict()  # 每个任务的结果

        for task_i, (task, val_loader) in enumerate(zip(self.task_ids, self.val_loaders)):
            nc = self.model_manager.data_dict["nc"][task_i]  # 获取任务的类别数量

            # 计算mAP和其他指标
            results, maps, _ = val.run(
                self.model_manager.data_dict,
                batch_size=max(self.batch_size) if isinstance(self.batch_size, list) else self.batch_size,
                imgsz=self.imgsz,
                model=ema.ema,
                single_cls=self.opt.single_cls,
                dataloader=val_loader,
                save_dir=self.model_manager.save_dir,
                verbose=nc < 50 and final_epoch and not self.opt.evolve,
                plots=plots,
                compute_loss=self.compute_loss,
                task_id=task,
                task_ind=task_i,
            )
            results_per_task[task] = results  # 存储每个任务的结果
            self.maps_per_task[task] = maps  # 存储每个任务的mAP

            # 更新最佳mAP
            fi = fitness(np.array(results).reshape(1, -1))  # 计算综合指标 [P, R, mAP@.5, mAP@.5-.95]
            fitness_per_task[task] = fi  # 存储任务的fitness
            if fi > self.best_fitness_per_task[task]:
                self.best_fitness_per_task[task] = fi  # 更新最佳fitness

                # 保存最佳任务模型
                if (not self.opt.nosave) or final_epoch:  # 如果需要保存
                    # 注意：仅用于模型追踪，不用于恢复训练
                    self.model_manager.save_best_task_model(
                        task,
                        epoch,
                        self.best_fitness_per_task,  # 不相关的任务fitness
                        self.best_fitness,  # 不相关的全局fitness
                        model,
                        ema,
                        self.get_optimizer_dict(),
                    )

            # 记录每个任务的验证日志
            self.model_manager.val_log(task, results, epoch, is_best=self.best_fitness_per_task[task] == fi)

        # 计算所有任务的平均fitness
        self.last_fitness = np.mean(list(fitness_per_task.values()))  # 计算当前fitness
        print("Cur fitness:", fitness_per_task, self.last_fitness)
        if self.last_fitness > self.best_fitness:
            self.best_fitness = self.last_fitness  # 更新全局最佳fitness

        # 保存模型
        if (not self.opt.nosave) or final_epoch:  # 如果需要保存
            is_best = self.best_fitness == self.last_fitness and not self.opt.evolve
            self.model_manager.save_model(
                epoch,
                self.best_fitness_per_task,
                self.best_fitness,
                model,
                ema,
                self.get_optimizer_dict(),
                is_best,
            )

        return results_per_task  # 返回每个任务的验证结果

