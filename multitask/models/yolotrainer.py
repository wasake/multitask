import argparse
import logging
import os
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# 提高 Ultralytics 日志级别，禁止CRITICAL，只显示警告或错误， WARNINGorERROR
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)

# 设置全局变量
LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).absolute().parents[1]

class YOLOv8Trainer:
    def __init__(self, hyp, opt, device):
        """
        初始化YOLOv8训练器
        :param hyp: 超参数字典或路径
        :param opt: 命令行参数选项
        :param device: 训练设备
        """
        self.opt = opt
        self.device = device
        self.cuda = device.type != "cpu"
        
        # 加载超参数
        if isinstance(hyp, str):
            with open(hyp, 'r') as f:
                self.hyp = yaml.safe_load(f)
        else:
            self.hyp = hyp
            
        # 加载YOLOv8模型
        self.model = YOLO(opt.weights if opt.weights else opt.cfg)
        
        # 保存目录（Windows兼容）
        self.save_dir = Path(opt.project) / opt.name
        self.save_dir = Path(increment_path(self.save_dir, exist_ok=opt.exist_ok, mkdir=True))
        
        # 数据配置
        with open(opt.data, 'r') as f:
            self.data_cfg = yaml.safe_load(f)
        self.num_tasks = len(self.data_cfg.get('tasks', [1]))  # 默认单任务

    def train_one_epoch(self, epoch):
        """
        训练一个epoch
        """
        
        # YOLOv8的训练接口
        results = self.model.train(
            data=self.opt.data,
            epochs=1,
            batch=self.opt.batch_size,
            imgsz=self.opt.imgsz,
            device=self.device,
            workers=0,  # Windows下设为0，避免多进程问题
            verbose=False, # 会产生大量原生日志
            project=str(self.save_dir),
            name=f'epoch_{epoch}',
            exist_ok=True
        )
        
        # 提取验证指标
        precision = results.results_dict['metrics/precision(B)']
        recall = results.results_dict['metrics/recall(B)']
        mAP50 = results.results_dict['metrics/mAP50(B)']
        mAP = results.results_dict['metrics/mAP50-95(B)']
        
        # 训练损失需要从日志中手动提取，这里假设为 0
        total_loss = 0  # 请根据日志或回调替换为实际值
        val_loss = 0    # 同上，需通过回调或日志获取
        
        LOGGER.info(f'Epoch {epoch} completed - Precision: {precision:.4f}, Recall: {recall:.4f}, '
                    f'mAP@0.5: {mAP50:.4f}, mAP@0.5:0.95: {mAP:.4f}')
        return total_loss

    def train(self, pt_path: str, epochs: int):
        """
        训练模型
        :param pt_path: 当前模型的 .pt 文件路径
        :param epochs: 训练的 epoch 数
        :return: 训练后的 .pt 文件路径
        """
        self.model.load(pt_path)
        start_epoch = 0
        if self.opt.resume:
            ckpt = self.opt.resume if isinstance(self.opt.resume, str) else str(self.save_dir / 'last.pt')
            if os.path.isfile(ckpt):
                self.model.load(ckpt)
                start_epoch = int(ckpt.split('_')[-1].split('.')[0]) + 1
                LOGGER.info(f"Resuming training from {ckpt} at epoch {start_epoch}")
        
        t0 = time.time()
        for epoch in range(start_epoch, start_epoch + epochs):
            avg_loss = self.train_one_epoch(epoch)
            self.model.save(self.save_dir / f'yolov8_epoch_{epoch}.pt')
            # LOGGER.info(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        final_pt_path = str(self.save_dir / f'yolov8_epoch_{start_epoch + epochs - 1}.pt')
        LOGGER.info(f"Training completed in {(time.time() - t0) / 3600:.3f} hours. Final model saved to {final_pt_path}")
        if self.cuda:
            torch.cuda.empty_cache()
        return final_pt_path

def increment_path(path, exist_ok=False, mkdir=False):
    """在Windows/Linux上递增路径，如 runs/exp -> runs/exp1"""
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = [x for x in path.parent.glob(path.name + '*') if x.is_dir()]
        matches = [int(x.name.split(path.name)[-1]) for x in dirs if x.name.split(path.name)[-1].isdigit()]
        i = max(matches, default=0) + 1 if matches else 1
        path = path.parent / f"{path.name}{i}{suffix}"
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path (optional if weights provided)')
    parser.add_argument('--data', type=str, default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='train image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--project', default=str(ROOT / 'runs/train'), help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='cuda or cpu')
    return parser.parse_args()

def main(opt):
    logging.basicConfig(level=logging.INFO)
    device = torch.device(opt.device)
    trainer = YOLOv8Trainer(opt.hyp, opt, device)
    trainer.train()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
