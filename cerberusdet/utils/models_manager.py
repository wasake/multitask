import logging
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.utils.checks import check_file
from cerberusdet.utils.ckpt_utils import dict_to_cerber, intersect_dicts
from cerberusdet.utils.general import check_dataset, colorstr, labels_to_class_weights, strip_optimizer
from cerberusdet.utils.mlflow_logging import MLFlowLogger
from cerberusdet.utils.plots import plot_images
from cerberusdet.utils.torch_utils import (
    ModelEMA,
    de_parallel,
    get_hyperparameter,
    is_parallel,
    model_info,
    set_hyperparameter,
    torch_distributed_zero_first,
)
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """
    负责模型的加载、保存、日志记录和训练过程中的图像绘制。
    """

    def __init__(self, hyp, opt, rank, local_rank):
        # 创建保存目录
        save_dir = opt.save_dir
        self.save_dir = Path(save_dir)
        wdir = self.save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)  # 创建目录
        self.last = wdir / "last.pt"  # 模型保存路径：最后的模型
        self.best = wdir / "best.pt"  # 模型保存路径：最佳模型
        self.results_file = self.save_dir / "results.txt"  # 训练结果保存文件

        self.opt = opt  # 存储训练参数
        self.rank = rank  # 分布式训练中的进程编号

        # 保存运行设置
        if not self.opt.evolve:
            with open(self.save_dir / "opt.yaml", "w") as f:
                yaml.safe_dump(vars(opt), f, sort_keys=False)  # 将配置保存为YAML文件

        # 获取超参数配置
        hyp = self.get_hyp(hyp)

        # 加载数据配置文件，包含训练、验证数据路径，类别数量和类别名称等
        with open(opt.data) as f:
            data_dict = yaml.safe_load(f)  # 解析数据字典

        # 在本地进程中检查数据集的有效性
        with torch_distributed_zero_first(local_rank):
            check_dataset(data_dict)  # 验证数据集路径是否存在

        # 判断数据是否为多任务
        if isinstance(data_dict["train"], list):
            self.num_tasks = len(data_dict["train"])  # 任务数
            self.task_ids = data_dict["task_ids"]  # 任务ID
        else:
            self.num_tasks = 1  # 单任务训练
            data_dict["train"] = [data_dict["train"]]  # 将训练数据包装成列表
            assert not isinstance(data_dict["val"], list) or len(data_dict["val"]) == 1  # 验证数据应为单任务
            data_dict["val"] = [data_dict["val"]]  # 将验证数据包装成列表
            if data_dict.get("task_ids") is None or len(data_dict["task_ids"]) != 1:
                data_dict["task_ids"] = ["detection"]  # 如果没有task_ids，默认任务为检测任务

            self.task_ids = data_dict["task_ids"]  # 获取任务ID

        # 确保任务ID唯一，任务数量正确
        assert len(np.unique(self.task_ids)) == self.num_tasks

        # 初始化日志记录器
        if rank in [-1, 0]:  # 只有主进程（rank为-1或0）才初始化日志
            self.loggers = self.get_loggers(hyp)

        # 获取模型训练的初始权重和训练轮数
        weights, epochs = opt.weights, opt.epochs

        # 对每个任务进行配置
        for i in range(self.num_tasks):
            task_nc = int(data_dict["nc"]) if not isinstance(data_dict["nc"], list) else int(data_dict["nc"][i])  # 获取任务类别数量
            task_nc = 1 if self.opt.single_cls else task_nc  # 如果是单类任务，则类别数设置为1

            # 获取任务类别名称
            task_names = data_dict["names"] if not isinstance(data_dict["nc"], list) else data_dict["names"][i]
            task_names = ["item"] if self.opt.single_cls and len(task_names) != 1 else task_names  # 单类任务的类别名称设置为"item"

            # 如果数据字典中没有多任务设置，则将数据字典包装成多任务格式
            if not isinstance(data_dict["nc"], list):
                data_dict["nc"] = [task_nc]  # 将类别数包装成列表
                data_dict["names"] = [task_names]  # 将任务名称包装成列表
            else:
                data_dict["nc"][i] = task_nc  # 更新该任务的类别数
                data_dict["names"][i] = task_names  # 更新该任务的类别名称

        # 保存数据字典
        self.data_dict = data_dict
        self.ckpt = None  # 检查点
        self.weights = weights  # 权重
        self.hyp = hyp  # 超参数
        self.epochs = epochs  # 训练轮数

        # 打印超参数信息
        LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in self.hyp.items()))

    def get_hyp(self, hyp, dump_name="hyp.yaml"):
        """
        获取超参数配置。如果超参数是文件路径，则加载文件内容。
        如果超参数已经是字典，则直接返回。

        参数:
            hyp (str or dict): 超参数配置文件路径或超参数字典
            dump_name (str): 保存超参数的文件名，默认为"hyp.yaml"

        返回:
            dict: 超参数字典
        """
        # 如果超参数是文件路径，则加载该文件
        if isinstance(hyp, str):
            hyp = check_file(hyp)

        # 如果没有进化训练，则保存超参数配置到指定文件
        if not self.opt.evolve:
            with open(self.save_dir / dump_name, "w") as f:
                yaml.safe_dump(hyp, f, sort_keys=False)  # 将超参数保存为YAML文件

        # 如果超参数是文件路径，则加载文件内容
        if isinstance(hyp, str):
            with open(hyp) as f:
                hyp = yaml.safe_load(f)  # 加载YAML文件并解析为字典

        return hyp

    def fill_tasks_parameters(self, nl, imgsz, model, datasets, device):
        """
        填充每个任务的超参数，包括类别权重、边界框损失权重、分类损失权重等。

        参数:
            nl (int): 网络层数
            imgsz (int): 图像大小
            model (nn.Module): 模型
            datasets (list): 数据集列表，每个任务一个数据集
            device (torch.device): 设备
        """
        model.names = dict()  # 初始化模型的类别名称字典
        model.class_weights = dict()  # 初始化模型的类别权重字典

        # 对每个任务进行超参数设置
        for task_i, (task, dataset) in enumerate(zip(self.task_ids, datasets)):
            nc = self.data_dict["nc"][task_i]  # 获取当前任务的类别数

            # 获取当前任务的边界框损失权重和分类损失权重
            box_w = get_hyperparameter(self.hyp, "box", task_i, task)
            cls_w = get_hyperparameter(self.hyp, "cls", task_i, task)

            # 按照层数和图像大小对权重进行缩放
            box_w *= 3.0 / nl  # 缩放边界框损失权重
            cls_w *= (imgsz / 640) ** 2 * 3.0 / nl  # 缩放分类损失权重

            # 更新超参数字典中的边界框和分类损失权重
            set_hyperparameter(self.hyp, "box", box_w, task_i, task)
            set_hyperparameter(self.hyp, "cls", cls_w, task_i, task)

            # 如果数据集不是子集，则计算并设置类别权重
            if not isinstance(dataset, torch.utils.data.Subset):
                model.class_weights[task] = (
                        labels_to_class_weights(dataset.labels, nc).to(device) * nc
                )  # 计算并保存类别权重

            # 设置模型的类别名称
            model.names[task] = self.data_dict["names"][task_i]

        # 将任务的类别数信息添加到模型
        model.nc = dict()
        for task, nc in zip(self.task_ids, self.data_dict["nc"]):
            model.nc[task] = nc  # 为每个任务设置类别数

        model.hyp = self.hyp  # 将超参数字典赋值给模型

        # 如果是多GPU并行训练，则将超参数信息复制到每个GPU的模型中
        if is_parallel(model):
            model.yaml = de_parallel(model).yaml
            model.stride = de_parallel(model).stride
            de_parallel(model).nc = model.nc
            de_parallel(model).hyp = self.hyp  # 将超参数赋值给并行模型

    def from_ckpt(self, ckpt, model, exclude=[]):
        """
        从检查点（ckpt）加载模型权重。

        参数:
            ckpt (str or dict): 检查点文件路径或检查点字典
            model (nn.Module): 需要加载权重的模型
            exclude (list): 排除的层名列表，默认空列表

        返回:
            tuple: 加载的状态字典和加载状态标志（True表示加载成功，False表示未加载）
        """
        # 如果检查点是字典并且包含"model"键，则获取模型的状态字典
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"].float().state_dict()  # 转换为FP32
        elif isinstance(ckpt, dict):
            state_dict = ckpt  # 如果是字典，则直接使用
        else:
            state_dict = ckpt.state_dict()  # 如果是模型对象，则获取其状态字典
        loaded = False  # 初始化加载标志

        # 如果加载的是来自YOLOv5的权重（含有"blocks."），则进行权重格式转换
        if "blocks." in list(model.state_dict().keys())[0] and "blocks." not in list(state_dict.keys())[0]:
            state_dict = dict_to_cerber(state_dict, model)  # 转换权重格式
            loaded = True  # 设置加载标志为True

            # 交集加载模型权重
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
            model.load_state_dict(state_dict, strict=False)  # 加载权重
            LOGGER.info("Transferred %g/%g items" % (len(state_dict), len(model.state_dict())))  # 打印加载报告

        return state_dict, loaded  # 返回加载的状态字典和加载状态

    def load_model(self, cfg, device, verbose=True):
        """
        加载模型及其权重。如果有预训练权重，则加载它们；否则初始化新的模型。

        参数:
            cfg (str or dict): 模型配置文件路径或配置字典
            device (torch.device): 设备
            verbose (bool): 是否打印详细信息，默认为True

        返回:
            model: 已加载的模型
            ema: 指数移动平均（EMA）模型（如果使用）
        """
        pretrained = self.weights.endswith(".pt")  # 判断是否是预训练权重文件

        loaded = False  # 初始化加载标志
        if self.rank in [-1, 0]:
            print(self.hyp)  # 打印超参数

        if pretrained:  # 如果有预训练权重
            LOGGER.info(f"Trying to restore weights from {self.weights} ...")

            self.ckpt = torch.load(self.weights, map_location=device)  # 加载检查点
            exclude = ["anchor"] if (cfg or self.hyp.get("anchors")) and not self.opt.resume else []  # 排除的键

            # 初始化模型
            model = CerberusDet(
                task_ids=self.task_ids,
                nc=self.data_dict["nc"],
                cfg=cfg or self.ckpt["model"].yaml,
                ch=3,
                verbose=verbose,
            ).to(device)

            # 从检查点加载模型
            state_dict, loaded = self.from_ckpt(self.ckpt, model, exclude)
        else:  # 没有预训练权重，则初始化新的模型
            model = CerberusDet(
                task_ids=self.task_ids,
                nc=self.data_dict["nc"],
                cfg=cfg,
                ch=3,
                verbose=verbose,
            ).to(device)

        # 在模型分裂前尝试加载预训练权重
        if model.yaml.get("cerber") and len(model.yaml["cerber"]):
            cerber_schedule = model.yaml["cerber"]
            if self.rank in [-1, 0] and self.loggers["mlflow"]:
                self.loggers["mlflow"].log_params({"cerber": cerber_schedule})
            model.sequential_split(deepcopy(cerber_schedule), device)
            if verbose and self.rank in [-1, 0]:
                print(model.info())

        if verbose and self.rank in [-1, 0]:
            model_info(model)

        if pretrained and not loaded:
            # 如果从预训练权重加载失败，则重新加载权重
            state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)
            model.load_state_dict(state_dict, strict=False)
            LOGGER.info(
                "Transferred %g/%g items from %s" % (len(state_dict), len(model.state_dict()), self.weights)
            )

        # 冻结模型参数
        freeze = []  # 要冻结的参数列表
        freeze_model(model, freeze)

        # 初始化EMA模型
        ema = ModelEMA(model) if self.rank in [-1, 0] else None

        # 加载EMA权重和训练结果
        if self.ckpt:
            if ema and self.ckpt.get("ema"):
                LOGGER.info("Loading ema from ckpt..")
                ema.ema.load_state_dict(self.ckpt["ema"].float().state_dict())
                ema.updates = self.ckpt["updates"]

            if self.ckpt.get("training_results") is not None:
                self.results_file.write_text(self.ckpt["training_results"])

            # 获取训练开始的epoch
            start_epoch = self.ckpt.get("epoch", -1) + 1
            if self.opt.resume:
                assert start_epoch > 0, "%s training to %g epochs is finished, nothing to resume." % (
                    self.weights,
                    self.epochs,
                )

            if self.epochs < start_epoch:
                LOGGER.info(
                    "%s has been trained for %g epochs. Fine-tuning for %g additional epochs."
                    % (self.weights, self.ckpt["epoch"], self.epochs)
                )
                self.epochs += self.ckpt["epoch"]

        return model, ema

    def save_model(
            self,
            epoch,
            best_fitness_per_task,
            best_fitness,
            model,
            ema,
            optimizer_state_dict,
            is_best=False,
    ):
        """
        保存模型的状态，包括模型、EMA权重、优化器状态等。

        参数:
            epoch (int): 当前训练的epoch
            best_fitness_per_task (dict): 每个任务的最佳适应度
            best_fitness (float): 所有任务的最佳适应度
            model (nn.Module): 当前模型
            ema (ModelEMA): 指数移动平均模型
            optimizer_state_dict (dict): 优化器状态字典
            is_best (bool): 是否是最佳模型，默认为False
        """
        if ema:
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        ckpt = self._get_ckpt_to_save(epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict)

        # 保存检查点
        torch.save(ckpt, self.last)
        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_model(str(self.last))

        if is_best:  # 如果是最佳模型，则保存到最佳模型路径
            torch.save(ckpt, self.best)
            if self.loggers["mlflow"]:
                self.loggers["mlflow"].log_model(str(self.best))

    def save_best_task_model(
            self, task_name, epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict
    ):
        """
        保存最佳任务模型。

        参数:
            task_name (str): 任务名称
            epoch (int): 当前训练的epoch
            best_fitness_per_task (dict): 每个任务的最佳适应度
            best_fitness (float): 所有任务的最佳适应度
            model (nn.Module): 当前模型
            ema (ModelEMA): 指数移动平均模型
            optimizer_state_dict (dict): 优化器状态字典
        """
        ckpt = self._get_ckpt_to_save(epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict)

        best_path = self.save_dir / "weights" / f"{task_name}_best.pt"
        torch.save(ckpt, best_path)
        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_model(str(best_path))

    def _get_ckpt_to_save(self, epoch, best_fitness_per_task, best_fitness, model, ema, optimizer_state_dict):
        """
        获取需要保存的检查点信息。

        参数:
            epoch (int): 当前训练的epoch
            best_fitness_per_task (dict): 每个任务的最佳适应度
            best_fitness (float): 所有任务的最佳适应度
            model (nn.Module): 当前模型
            ema (ModelEMA): 指数移动平均模型
            optimizer_state_dict (dict): 优化器状态字典

        返回:
            dict: 检查点字典，包含训练结果、模型、EMA、优化器等信息
        """
        training_results = self.results_file.read_text() if self.results_file.exists() else None
        ckpt = {
            "epoch": epoch,
            "best_fitness_per_task": best_fitness_per_task,
            "best_fitness": best_fitness,
            "training_results": training_results,
            "model": deepcopy(de_parallel(model)).half(),
            "ema": deepcopy(ema.ema).half(),
            "updates": ema.updates,
            "optimizer": optimizer_state_dict,
        }
        return ckpt

    def strip_optimizer(self):
        """
        删除优化器状态信息，减少模型文件大小。
        """
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # 删除优化器状态

    def log_models(self):
        """
        将模型记录到MLFlow或TensorBoard中。
        """
        if self.loggers["mlflow"] and not self.opt.evolve:
            if self.best.exists():
                self.loggers["mlflow"].log_model(str(self.best))
            if self.last.exists():
                self.loggers["mlflow"].log_model(str(self.last))

    def train_log(
            self,
            task,
            lr,
            mloss,
            epoch,
            last_stat,
            scale_t=None,
            tag_prefix="",
            tags=("box_loss", "obj_loss", "cls_loss", "lr0", "lr1", "lr2"),
    ):
        """
        记录训练信息到TensorBoard、MLFlow和本地文件。

        参数:
            task (str): 任务名称
            lr (list): 学习率列表
            mloss (list): 损失值列表
            epoch (int): 当前epoch
            last_stat (str): 最近的统计信息
            scale_t (float or None): 缩放因子
            tag_prefix (str): 标签前缀
            tags (tuple): 标签列表
        """
        mlflow_metrics = {}
        cnt = 0

        # 记录损失和学习率
        for full_prefix, param_group in zip(
                [f"{tag_prefix}train/{task}/", f"{tag_prefix}x/{task}/"], [list(mloss[:-1]), lr]
        ):
            n_group_params = len(param_group)
            group_tags = tags[cnt: (cnt + n_group_params)]
            for x, tag in zip(param_group, group_tags):
                full_tag = f"{full_prefix}{tag}"
                if self.loggers["tb"]:
                    self.loggers["tb"].add_scalar(full_tag, x, epoch)  # 记录到TensorBoard
                if self.loggers["mlflow"]:  # 记录到MLFlow
                    mlflow_metrics[full_tag.replace("/", "_")] = (
                        float(x.cpu().numpy()) if isinstance(x, torch.Tensor) else x
                    )
            cnt += n_group_params

        if scale_t:
            tag = f"{tag_prefix}{task}/scale"
            if self.loggers["tb"]:
                self.loggers["tb"].add_scalar(tag, scale_t, epoch)  # TensorBoard
            if self.loggers["mlflow"]:  # MLFlow
                mlflow_metrics[tag.replace("/", "_")] = (
                    float(scale_t.cpu().numpy()) if isinstance(scale_t, torch.Tensor) else scale_t
                )

        with open(self.results_file, "a") as f:
            f.write(f"Train {task}: " + last_stat + "\n")  # 记录到文件

        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_metrics(mlflow_metrics, step=epoch)

    def val_log(self, task, results, epoch, is_best):
        """
        记录验证信息到TensorBoard、MLFlow和本地文件。

        参数:
            task (str): 任务名称
            results (list): 验证结果
            epoch (int): 当前epoch
            is_best (bool): 是否为最佳模型
        """
        with open(self.results_file, "a") as f:
            f.write(f"Val {task}: " + "%10.4g" * 7 % results + "\n")  # 记录到文件

        mlflow_metrics = {}

        # 记录验证结果
        tags = [
            f"metrics/{task}/precision",
            f"metrics/{task}/recall",
            f"metrics/{task}/mAP_0.5",
            f"metrics/{task}/mAP_0.5:0.95",
            f"val/{task}/box_loss",
            f"val/{task}/obj_loss",
            f"val/{task}/cls_loss",  # 验证损失
        ]

        for x, tag in zip(list(results), tags):
            if self.loggers["tb"]:
                self.loggers["tb"].add_scalar(tag, x, epoch)  # 记录到TensorBoard
            if self.loggers["mlflow"]:  # 记录到MLFlow
                mlflow_metrics[tag.replace("/", "_").replace(":", "_")] = (
                    float(x.cpu().numpy()) if isinstance(x, torch.Tensor) else x
                )

        if self.loggers["mlflow"]:
            self.loggers["mlflow"].log_metrics(mlflow_metrics, step=epoch)

    def plot_train_images(self, ni, task, batch, model):
        """
        在训练过程中保存一批图像的可视化结果。

        参数:
            ni (int): 当前训练步数
            task (str): 当前任务名称
            batch (dict): 当前批次的数据
            model (nn.Module): 当前模型
        """
        imgs = batch["img"]  # 获取图像数据
        if ni < 3:  # 仅在前3步训练中进行可视化
            # 保存图像并记录到文件
            plot_images(
                images=batch["img"],
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),  # 去掉多余的维度
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}_{task}.jpg",  # 保存文件路径
                mlflow_logger=self.loggers["mlflow"],  # 记录到MLFlow
            )

            # 如果使用TensorBoard，记录图像到TensorBoard
            if self.loggers["tb"] and ni == 0:  # 仅在第一个步数记录
                if not self.opt.sync_bn:  # 如果不使用同步批归一化
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")  # 忽略JIT追踪警告
                        if hasattr(model, "set_task"):
                            model.set_task(task)  # 设置当前任务
                        # 记录模型的计算图到TensorBoard
                        self.loggers["tb"].add_graph(torch.jit.trace(model, imgs[0:1], strict=False), [])

    def get_loggers(self, hyp, include=("tb",)):
        """
        初始化并返回日志记录器（TensorBoard 和 MLFlow）。

        参数:
            hyp (dict): 超参数字典
            include (tuple): 包含的日志记录器类型，默认为("tb",)

        返回:
            dict: 包含 TensorBoard 和 MLFlow 日志记录器的字典
        """
        loggers = {"tb": None, "mlflow": None}  # 初始化日志记录器字典

        # 如果没有进化训练且需要TensorBoard
        if not self.opt.evolve and "tb" in include:
            prefix = colorstr("tensorboard: ")
            LOGGER.info(
                f"{prefix}Start with 'tensorboard --logdir {self.opt.project}', view at http://localhost:6006/")
            loggers["tb"] = SummaryWriter(str(self.save_dir))  # 创建TensorBoard记录器

        # 如果配置了MLFlow URL，使用MLFlow记录器
        if self.opt.mlflow_url:
            loggers["mlflow"] = MLFlowLogger(self.opt, hyp)
        else:
            LOGGER.info("MLFlow logger will not be used")
            loggers["mlflow"] = None  # 如果没有配置，MLFlow记录器为None

        self.opt.hyp = hyp  # 将超参数存储到选项中

        return loggers

def freeze_model(model, freeze):
    """
    冻结模型的部分层，使其参数不进行更新。

    参数:
        model (nn.Module): 要冻结的模型
        freeze (list): 需要冻结的层的名称列表
    """
    for k, v in model.named_parameters():  # 遍历模型的所有参数
        if v.requires_grad:
            v.requires_grad = True  # 默认训练所有层
        if any(x in k for x in freeze):  # 如果参数名包含需要冻结的层
            print("freezing %s" % k)  # 输出冻结层的名称
            v.requires_grad = False  # 冻结该层，不更新权重

