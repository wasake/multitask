import argparse
import logging
import os
import time
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import yaml
from cerberusdet import val  # 导入验证相关模块
from cerberusdet.data.dataloaders import _create_dataloader  # 数据加载器函数
from cerberusdet.data.datasets import LoadImagesAndLabels  # 数据集加载器
from cerberusdet.evolvers.predefined_evolvers import EVOLVER_TYPES  # 演化器类型
from cerberusdet.models.cerberus import CerberusDet  # CerberusDet模型
from cerberusdet.models.experimental import attempt_load  # 尝试加载模型函数
from cerberusdet.trainers.averaging import Averaging  # 用于平均训练结果的训练器
from cerberusdet.utils.checks import check_file, check_git_status, check_requirements  # 检查文件、git状态和依赖
from cerberusdet.utils.general import check_img_size, colorstr, get_latest_run, increment_path, init_seeds, set_logging  # 通用工具函数
from cerberusdet.utils.models_manager import ModelManager  # 模型管理器
from cerberusdet.utils.plots import plot_labels  # 标签绘图工具
from cerberusdet.utils.torch_utils import select_device  # 选择设备（CPU或GPU）
from cerberusdet.utils.train_utils import create_data_loaders, get_init_metrics_per_task  # 训练相关工具函数
from loguru import logger  # 用于日志记录的库
from torch.nn.parallel import DistributedDataParallel as DDP  # 用于多GPU训练的分布式数据并行

torch.backends.cudnn.enabled = False  # 禁用cuDNN，以确保在没有GPU的环境下也能运行
warnings.filterwarnings(
    "ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated"
)  # 忽略特定的警告信息

# 设置全局变量
LOGGER = logging.getLogger(__name__)  # 获取当前模块的日志记录器
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # 本地进程的编号，主要用于多GPU训练
RANK = int(os.getenv("RANK", -1))  # 全局进程的编号
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # 总的进程数
ROOT = Path(__file__).absolute().parents[1]  # 获取项目的根目录，通常是脚本的父目录
def train(
    hyp,  # 超参数字典或路径，包含训练时所需的超参数
    opt,
    device,
    train_dataset: Optional[List[LoadImagesAndLabels]] = None,  # 可选的自定义训练数据集
    val_dataset: Optional[List[LoadImagesAndLabels]] = None,  # 可选的自定义验证数据集
):
    """
    训练模型的主函数，返回每个任务的评估结果

    :param hyp: 超参数字典或路径，包含了如学习率、批次大小等训练参数
    :param opt: 命令行参数选项，包含如训练周期、GPU设备等设置
    :param device: 当前训练设备（CPU或GPU）
    :param train_dataset: 训练数据集（可选），如果提供将使用提供的训练数据
    :param val_dataset: 验证数据集（可选），如果提供将使用提供的验证数据
    :return: 每个任务的评估结果，包含精度（P），召回率（R），mAP 等
    """

    # 获取是否进行超参数演化的标志以及配置文件路径
    evolve, cfg = opt.evolve, opt.cfg

    # 配置标志，是否绘制训练过程中的图表
    plots = not evolve
    cuda = device.type != "cpu"  # 是否使用GPU
    init_seeds(1 + RANK)  # 初始化随机种子，确保结果可复现

    # 创建模型管理器实例，负责管理模型及其相关资源
    model_manager = ModelManager(hyp, opt, RANK, LOCAL_RANK)
    save_dir = model_manager.save_dir  # 获取模型保存路径
    hyp = model_manager.hyp  # 获取超参数配置

    # 加载模型和EMA（指数移动平均）权重
    verbose_model = not evolve  # 是否打印详细的模型信息
    model, ema = model_manager.load_model(cfg, device, verbose_model)  # 加载模型并返回EMA权重

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    if isinstance(model, CerberusDet):
        nl = None
        for task_name in model.heads:
            head = model.get_head(task_name)
            if nl is None:
                nl = head.nl
            else:
                assert nl == head.nl
            assert hasattr(head, "stride")
    else:
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs)  # 验证输入图像大小是否为grid大小的倍数

    # 数据加载器（train_loader 和 val_loader）
    if train_dataset is None:
        # 如果没有提供自定义数据集，使用默认的数据加载器
        train_loader, val_loader, dataset, _ = create_data_loaders(
            model_manager.data_dict, RANK, WORLD_SIZE, opt, hyp, gs, imgsz, balanced_sampler=True
        )
    else:
        # 如果提供了自定义数据集，创建训练和验证数据加载器
        train_loader = []
        dataset = train_dataset
        for ii, train_task_dataset in enumerate(dataset):
            train_task_dataset.update_hyp(hyp)  # 更新超参数
            train_task_dataloader = _create_dataloader(
                train_task_dataset,
                workers=opt.workers,
                batch_size=opt.batch_size[ii] if isinstance(opt.batch_size, list) else opt.batch_size,
                rank=RANK,
                use_balanced_sampler=True,
            )
            train_loader.append(train_task_dataloader)

        val_loader = []
        for val_task_dataset in val_dataset:
            val_task_dataloader = _create_dataloader(
                val_task_dataset,
                workers=opt.workers,
                batch_size=max(opt.batch_size) if isinstance(opt.batch_size, list) else opt.batch_size,
                rank=-1,
                use_balanced_sampler=False,
            )
            val_loader.append(val_task_dataloader)

    assert len(dataset) == model_manager.num_tasks  # 检查数据集数量与模型任务数量是否一致

    # 创建优化器
    trainer = Averaging(device, model, model_manager, train_loader, val_loader, dataset, imgsz, gs)

    # 恢复训练
    start_epoch = trainer.resume(model_manager.ckpt)

    # 数据并行（DP模式）警告
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logger.warning(
            "DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started."
        )
        model = torch.nn.DataParallel(model)

    # 使用同步批归一化（SyncBatchNorm）处理多GPU训练
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        if RANK in [-1, 0]:
            logger.info("Using SyncBatchNorm()")

    # 训练前的任务初始化和数据检查
    if RANK in [-1, 0] and not model_manager.opt.resume:
        for ii, (task_dataset, task_name) in enumerate(zip(dataset, model_manager.task_ids)):
            task_nc = model_manager.data_dict["nc"][ii]  # number of classes
            names = model_manager.data_dict["names"][ii]
            assert len(names) == task_nc, "%g names found for nc=%g dataset in %s" % (
                len(names),
                task_nc,
                opt.data,
            )  # 检查类名数量是否与类数匹配

            labels = np.concatenate(task_dataset.labels, 0)
            if plots:
                plot_labels(labels, names, save_dir, model_manager.loggers, name=task_name)

        model.half().float()  # 对模型进行半精度处理，提升计算效率

    # 在MLflow中记录模型和训练信息
    if RANK in [-1, 0] and model_manager.loggers["mlflow"] and not evolve:
        model_manager.loggers["mlflow"].save_artifacts(save_dir)

    # 校验数据标签的有效性
    for ii, (task_dataset, task_name) in enumerate(zip(dataset, model_manager.task_ids)):
        labels = np.concatenate(task_dataset.labels, 0)

        if labels.shape[1] == 6:
            mlc = labels[:, 0].max()  # 最大标签类别
        elif labels.shape[1] == 7:
            mlc = labels[:, 1].max()

        task_nc = model_manager.data_dict["nc"][ii]
        assert mlc < task_nc, "Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g" % (
            mlc,
            task_nc,
            opt.data,
            task_nc - 1,
        )

    # DDP模式下的处理
    if cuda and RANK != -1:
        logger.info(f"Using DDP on gpu {LOCAL_RANK}")
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)

    # 填充模型参数（根据数据集任务数量）
    model_manager.fill_tasks_parameters(nl, imgsz, model, trainer.dataset, device)

    # 开始训练
    t0 = time.time()

    # 初始化每个任务的评估指标（P、R、mAP等）
    results_per_task = get_init_metrics_per_task(model_manager)

    # 设置损失函数
    trainer.set_loss(model)
    if isinstance(trainer.scheduler, dict):
        for _, scheduler in trainer.scheduler.items():
            scheduler.last_epoch = start_epoch - 1
    else:
        trainer.scheduler.last_epoch = start_epoch - 1

    # 输出训练配置
    if RANK in [-1, 0]:
        logger.info(
            f"Image sizes {imgsz} train, {imgsz} val\n"
            f"Using {train_loader[0].num_workers} dataloader workers\n"
            f"Logging results to {model_manager.save_dir}\n"
            f"Starting training for {model_manager.epochs} epochs..."
        )

    # 训练与验证过程
    for epoch in range(start_epoch, model_manager.epochs):
        prev_best_fitness = trainer.best_fitness
        trainer.train_epoch(model, ema, epoch, RANK, WORLD_SIZE)

        # DDP进程0或单GPU
        if RANK in [-1, 0]:
            results_per_task = trainer.val_epoch(model, ema, epoch, WORLD_SIZE)

            # 如果模型表现最佳且不进行演化，更新最佳模型
            if trainer.best_fitness > prev_best_fitness and not evolve and not opt.nosave:
                logger.info("Best model updated")

            stop = trainer.stopper(epoch=epoch, fitness=trainer.last_fitness)
            if stop:
                break

    # 训练结束后记录信息
    if RANK in [-1, 0]:
        logger.info(f"{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n")

        # 验证模型并记录评估结果
        if not evolve:
            for task_i, (task, val_loader) in enumerate(zip(trainer.task_ids, trainer.val_loaders)):
                ckpts = (
                    [model_manager.last, model_manager.best] if model_manager.best.exists() else [model_manager.last]
                )
                for m in ckpts:  # speed, mAP tests
                    results_per_task[task] = val.run(
                        model_manager.data_dict,
                        batch_size=max(trainer.batch_size) if isinstance(trainer.batch_size, list) else trainer.batch_size,
                        imgsz=imgsz,
                        model=attempt_load(m, device).half(),
                        single_cls=model_manager.opt.single_cls,
                        dataloader=val_loader,
                        save_dir=model_manager.save_dir,
                        plots=True,
                        verbose=True,
                        task_id=task,
                        task_ind=task_i,
                        labels_from_xml=opt.labels_from_xml,
                        use_multi_labels=opt.use_multi_labels,
                        use_soft_labels=opt.use_soft_labels,
                    )[0]

                    if model_manager.loggers["mlflow"] and m == model_manager.best:
                        model_manager.loggers["mlflow"].save_artifacts(model_manager.save_dir)

        # 清理优化器
        model_manager.strip_optimizer()
        model_manager.log_models()

        # 在MLflow中保存最终模型
        if model_manager.loggers["mlflow"]:
            if not evolve:
                model_manager.loggers["mlflow"].save_artifacts(model_manager.save_dir)
                if model_manager.best.exists():
                    half_best_model = attempt_load(model_manager.best, device).half()
                    model_manager.loggers["mlflow"].log_model_signature(half_best_model, imgsz, device, "best_model")
                    model_manager.loggers["mlflow"].log_best_model_md5(str(model_manager.best), "best_model")
            else:
                # 演化过程中的处理，这里不会保存模型
                pass
            model_manager.loggers["mlflow"].finish_run()

    # 清空CUDA缓存
    torch.cuda.empty_cache()
    return results_per_task, epoch


def parse_opt(known=False):
    """
    解析命令行参数的函数，用于训练过程中的配置设置。

    :param known: 是否允许解析未知的参数，默认为False。如果为True，则解析时会忽略未知的命令行参数。
    :return: 返回解析后的参数（一个命名空间对象）
    """
    parser = argparse.ArgumentParser()  # 创建命令行解析器

    # 添加命令行参数
    parser.add_argument("--weights", type=str, default="../pretrained/yolov8x_state_dict.pt",
                        help="initial weights path")  # 初始权重路径，默认是`pretrained/yolov8x_state_dict.pt`

    parser.add_argument("--cfg", type=str, default="../cerberusdet/models/yolov8x_voc_obj365.yaml",
                        help="model.yaml path")  # 模型配置文件路径，默认是`cerberusdet/models/yolov8x.yaml`

    parser.add_argument("--data", type=str, default="../data/voc_obj365_animals.yaml",
                        help="dataset.yaml path")  # 数据集配置文件路径，默认是`data/voc_obj365.yaml`

    parser.add_argument("--hyp", type=str, default="../data/hyps/hyp.cerber-voc_obj365.yaml",
                        help="hyperparameters path")  # 超参数配置文件路径，默认是`data/hyps/hyp.cerber-voc_obj365.yaml`

    parser.add_argument("--epochs", type=int, default=100)  # 训练的周期数，默认是100

    parser.add_argument("--batch-size", type=str, default='8',
                        help="batch size for one GPUs")  # 每个GPU的批次大小，默认是32

    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640,
                        help="train, val image size (pixels)")  # 训练和验证的图像大小，默认是640像素

    parser.add_argument("--resume", nargs="?", const=True, default=False,
                        help="resume most recent training")  # 是否恢复最新的训练状态，默认为False

    parser.add_argument("--nosave", action="store_true",
                        help="only save final checkpoint")  # 是否只保存最终的检查点，不保存中间结果

    parser.add_argument("--noval", action="store_true",
                        help="only validate final epoch")  # 是否只在最后一个周期进行验证

    parser.add_argument("--evolve", type=int, nargs="?", const=300,
                        help="evolve hyperparameters for x generations")  # 是否进行超参数演化，默认为300代

    parser.add_argument("--evolver", type=str, default="yolov5",
                        help="Evolve algo to use", choices=EVOLVER_TYPES + ["yolov5"])  # 使用的超参数演化算法，默认是`yolov5`

    parser.add_argument("--params_to_evolve", type=str, default=None,
                        help="Parameters to find separated by comma")  # 要演化的超参数，用逗号分隔

    parser.add_argument("--evolve_per_task", action="store_true",
                        help="whether to evolve params specified per task differently")  # 是否为每个任务指定不同的超参数演化

    parser.add_argument("--cache-images", action="store_true",
                        help="cache images for faster training")  # 是否缓存图像以加速训练

    parser.add_argument("--device", default="0",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")  # 使用的设备，支持指定CUDA设备（例如`0`，`0,1,2,3`）或使用CPU

    parser.add_argument("--sync-bn", action="store_true",
                        help="use SyncBatchNorm, only available in DDP mode")  # 是否使用同步批归一化，仅在DDP模式下可用

    parser.add_argument("--workers", type=int, default=16,
                        help="maximum number of dataloader workers")  # Dataloader的最大工作线程数，默认为16

    parser.add_argument("--project", default=str(ROOT / "runs/train"),
                        help="save to project/name")  # 保存路径，默认是`runs/train`目录

    parser.add_argument("--name", default="voc_voc",
                        help="save to project/name")  # 保存的实验名称，默认为`exp`

    parser.add_argument("--exist-ok", action="store_true",
                        help="existing project/name ok, do not increment")  # 如果目录已存在，是否不递增版本号

    parser.add_argument("--linear-lr", action="store_true",
                        help="linear LR")  # 是否使用线性学习率调整

    parser.add_argument("--experiment_name", type=str, default="cerberus_exp",
                        help="MlFlow experiment name")  # MLFlow实验名称，默认是`cerberus_exp`

    parser.add_argument("--patience", type=int, default=30,
                        help="EarlyStopping patience (epochs without improvement)")  # 提前停止的耐心值（在没有提升的情况下允许多少个周期）

    parser.add_argument("--mlflow-url", type=str, default='localhost',
                        help="Param for mlflow.set_tracking_uri(), may be 'local'")  # MLFlow的跟踪URI，默认为`None`

    parser.add_argument("--local-rank", type=int, default=-1,
                        help="DDP parameter, do not modify")  # DDP参数，用于分布式训练的本地进程编号

    parser.add_argument("--single-cls", action="store_true",
                        help="train multi-class data as single-class")  # 是否将多类数据当作单类数据进行训练

    parser.add_argument("--use-multi-labels", action="store_true",
                        help="Loading multiple labels for boxes, if available")  # 是否加载多个标签用于框，若有多个标签

    parser.add_argument("--use-soft-labels", action="store_true",
                        help="Class probability based on annotation votes")  # 是否使用软标签（基于注释投票的类别概率）

    parser.add_argument("--labels-from-xml", action="store_true",
                        help="Load labels from xml files")  # 是否从XML文件中加载标签

    parser.add_argument("--freeze-shared-till-epoch", type=int, default=0,
                        help="Freeze shared between all tasks params for first N epochs")  # 冻结共享参数，直到训练的第N个周期，默认为0

    parser.add_argument("--skip-batches", action="store_true",
                        help="skip batches of small datasets so that the model sees them only once per epoch.")  
    # 是否跳过小数据集的批次，每个周期只看到一次epoch

    # 解析参数，`parse_known_args()`允许解析未知参数
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt  # 返回解析后的参数对象


def main(opt):
    """
    主训练函数，管理训练的各个阶段，包括恢复训练、分布式训练、超参数演化等。

    :param opt: 训练时的所有超参数和配置选项，通常是通过命令行解析获得的。
    """
    set_logging(RANK)  # 设置日志记录（用于输出训练过程中的信息）

    # 如果是主进程（RANK为-1或0），输出训练的配置参数
    if RANK in [-1, 0]:
        print(colorstr("train: ") + ", ".join(f"{k}={v}" for k, v in vars(opt).items()))
        check_git_status()  # 检查当前Git仓库的状态，确保没有未提交的更改

    # 如果选择了恢复训练（opt.resume），且没有进行超参数演化（opt.evolve）
    if opt.resume and not opt.evolve:  # 恢复中断的训练
        # 获取恢复训练的检查点路径（如果是字符串类型，使用指定路径；否则获取最新的检查点路径）
        ckpt = (
            opt.resume if isinstance(opt.resume, str) else get_latest_run(opt.project)
        )
        logger.info(f"Resume from {ckpt}")  # 记录恢复训练的路径

        # 检查检查点文件是否存在
        assert os.path.isfile(ckpt), "ERROR: --resume checkpoint does not exist"

        # 从检查点的父目录中读取训练配置（opt.yaml文件），并更新opt参数
        with open(Path(ckpt).parent.parent / "opt.yaml") as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # 更新opt配置

        # 重新设置配置参数，确保恢复训练的状态
        opt.cfg, opt.weights, opt.resume = "", ckpt, True  # 清空cfg，设置恢复路径

        LOGGER.info(f"Resuming training from {ckpt}")
    else:
        # 如果没有恢复训练，则检查数据集、配置文件和超参数文件是否存在
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # 检查文件路径
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"  # 必须指定cfg或weights

        # 如果选择超参数演化（opt.evolve），设置项目名称并修改路径
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # 如果项目名称使用默认值`runs/train`，改为`runs/evolve`
                opt.project = str(ROOT / "runs/evolve")
            opt.name = f"{opt.evolver}_{opt.name}"  # 修改实验名称，加入演化算法名称

        # 设置是否覆盖已有的项目目录，并禁用恢复训练（opt.resume）
        opt.exist_ok, opt.resume = opt.resume, False  # 将resume的值传递给exist_ok，禁用恢复训练

        # 如果是主进程（RANK为-1或0），设置保存路径
        if RANK in [-1, 0]:
            # 创建保存目录，防止目录名重复，使用`increment_path`递增路径
            opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True))
        else:
            opt.save_dir = ""  # 如果是非主进程，保存路径为空

    # 如果批次大小是字符串类型（例如多个GPU的批次大小以逗号分隔），则将其转换为整数列表
    if isinstance(opt.batch_size, str):
        opt.batch_size = list(map(int, opt.batch_size.split(",")))  # 将批次大小按逗号分隔成整数列表
        if len(opt.batch_size) == 1:  # 如果只有一个批次大小，直接取第一个
            opt.batch_size = opt.batch_size[0]

    # DDP（分布式数据并行）模式
    if LOCAL_RANK != -1:
        from datetime import timedelta

        # 检查CUDA设备是否足够用于DDP训练
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)  # 设置当前使用的GPU
        device = torch.device("cuda", LOCAL_RANK)  # 创建一个CUDA设备对象，指定本地设备
        # 初始化分布式训练进程组，使用`nccl`后端（适用于NVIDIA GPU），如果不可用则使用`gloo`
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10000)
        )
    else:
        # 选择训练设备（CPU或GPU），如果指定了设备（opt.device），则使用它
        device = select_device(
            opt.device, batch_size=max(opt.batch_size) if isinstance(opt.batch_size, list) else opt.batch_size
        )

    # 正常训练
    if not opt.evolve:
        # 调用训练函数，开始训练
        train(opt.hyp, opt, device)

        # 如果有多个进程（分布式训练），主进程（RANK == 0）需要销毁进程组
        if WORLD_SIZE > 1 and RANK == 0:
            print("Destroying process group... ")
            dist.destroy_process_group()  # 销毁分布式进程组
            print("Done.")

    # 超参数演化（可选）
    else:
        from evolvers import RayEvolver, Yolov5Evolver

        # 根据选择的演化算法（`opt.evolver`），初始化相应的演化器
        if opt.evolver == "yolov5":
            evolver = Yolov5Evolver(opt, device)  # 使用Yolov5的演化器
        else:
            evolver = RayEvolver(opt, device)  # 使用Ray的演化器

        # 执行超参数演化
        evolver.run_evolution(train)  # 启动超参数演化过程，传入训练函数


def run(kwargs):
    # Usage: from cerberusdet import train; train.run(imgsz=640, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    new_pt_path = opt.save_dir + "/weights/best.pt"
    return new_pt_path


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
