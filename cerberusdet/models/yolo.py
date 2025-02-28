# 导入必要的库
import logging  # 用于记录日志
import math  # 数学函数库
import os  # 用于与操作系统交互（如路径、环境变量等）
import sys  # 与 Python 解释器交互，修改路径等
from copy import deepcopy  # 用于深拷贝对象
from pathlib import Path  # 用于处理路径操作

# 导入 PyTorch 相关库
import torch  # PyTorch 主库
import torch.nn as nn  # PyTorch 神经网络模块

# 导入 CerberusDet 模型和工具
from cerberusdet.models.common import (
    C2, C3, C3SPP, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, Concat, Contract, Conv, DWConv, Expand, Focus,
)
from cerberusdet.models.experimental import CrossConv, GhostBottleneck, GhostConv, MixConv2d  # 一些实验性的模块
from cerberusdet.utils.general import make_divisible  # 工具函数，用于使数值可被某个数整除
from cerberusdet.utils.plots import feature_visualization  # 可视化特征
from cerberusdet.utils.tal import dist2bbox, make_anchors  # 工具函数，用于计算目标框和生成锚框
from cerberusdet.utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, time_sync  # PyTorch 工具函数
from loguru import logger  # 更强大的日志库

# 尝试导入 thop 用于计算 FLOPs（每秒浮点运算次数），如果失败则设置为 None
try:
    import thop  # 用于计算FLOPs
except ImportError:
    thop = None

# 配置日志
LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # 获取本地的 GPU 设备编号
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))  # 获取总共的 GPU 数量
RANK = int(os.getenv("RANK", -1))  # 获取进程的全局编号

# 获取当前文件的绝对路径
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # 将上一级文件夹添加到路径中，目的是为了能够导入 cerberusdet 模块

# 定义 DFL 模块
class DFL(nn.Module):
    # DFL 模块用于目标框的距离变换
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)  # 创建卷积层，输出通道为1，不需要偏置，并且禁止训练
        x = torch.arange(c1, dtype=torch.float)  # 创建一个从 0 到 c1 的张量
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))  # 将卷积层的权重初始化为这个张量
        self.c1 = c1  # 保存输入通道数

    def forward(self, x):
        b, c, a = x.shape  # 获取输入张量的批次大小（b），通道数（c），和锚框数量（a）
        # 对输入张量进行变形后通过卷积层，最后返回处理后的目标框
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# 定义 Detect 模块，用于目标检测的头部
class Detect(nn.Module):
    # YOLOv8 检测头，用于目标检测模型
    dynamic = False  # 是否强制进行网格重建
    export = False  # 是否为导出模式
    shape = None  # 存储输入图像的尺寸
    anchors = torch.empty(0)  # 初始化锚框为空
    strides = torch.empty(0)  # 初始化步幅为空

    def __init__(self, nc=80, ch=()):  # 初始化检测层
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层的数量
        self.reg_max = 16  # DFL 的通道数（缩放系数，用于生成锚框）
        self.no = nc + self.reg_max * 4  # 每个锚框的输出数
        self.stride = torch.zeros(self.nl)  # 步幅数组，后续会计算

        # 创建检测模块的卷积层
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # 计算卷积的输入输出通道数
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()  # 如果 reg_max > 1 使用 DFL 否则使用恒等映射

    def forward(self, x):
        shape = x[0].shape  # 获取输入的尺寸
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)  # 将 cv2 和 cv3 的输出拼接起来
        if self.training:
            return x  # 在训练模式下返回中间结果
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 计算目标框和类别
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        # 初始化 Detect() 层的偏置
        m = self  # self.model[-1]  # Detect() 模块
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # 迭代 cv2 和 cv3 层
            a[-1].bias.data[:] = 1.0  # 盒子预测的偏置初始化为 1
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # 类别预测的偏置初始化
class Model(nn.Module):
    # 初始化模型的构造函数
    def __init__(self, cfg="yolov8x.yaml", ch=3, nc=None, without_head=False, _=None, verbose=True):
        super().__init__()

        # 判断传入的配置是字典还是文件路径，如果是文件路径则加载yaml文件
        if isinstance(cfg, dict):
            self.yaml = cfg  # 如果是字典，直接赋值给self.yaml
        else:  # 如果是文件路径，读取yaml文件
            import yaml  # 导入yaml库用于解析yaml文件

            self.yaml_file = Path(cfg).name  # 获取yaml文件的文件名
            with open(cfg) as f:  # 打开yaml文件
                self.yaml = yaml.safe_load(f)  # 使用yaml库加载yaml文件内容

        # 定义模型的输入通道数
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # 如果yaml配置中没有ch，使用默认值ch

        # 如果nc是一个列表且yaml中有nc配置，则覆盖yaml中的nc配置
        if isinstance(nc, list) and self.yaml.get("nc", None) is not None:
            self.yaml["nc"] = nc  # 覆盖yaml中的nc
        # 如果nc不是列表且yaml中的nc值为空或者nc与yaml中的nc不一致，则更新yaml中的nc
        elif nc and (self.yaml.get("nc") is None or nc != self.yaml["nc"]):
            if verbose and LOCAL_RANK in [0, -1]:
                LOGGER.info(f"Overriding model.yaml nc={self.yaml.get('nc')} with nc={nc}")
            self.yaml["nc"] = nc  # 更新yaml中的nc

        # 解析模型，返回模型层和需要保存的层
        self.model, self.save, self.saved_ch = parse_model(
            deepcopy(self.yaml), ch=[ch], without_head=without_head, verbose=verbose
        )  # 调用parse_model函数，生成模型和保存的层

        self.without_head = without_head  # 设置是否去掉头部（检测头）
        self.inplace = self.yaml.get("inplace", True)  # 获取yaml中是否启用inplace的配置

        # 如果没有去掉头部
        if not without_head:
            # 如果yaml中的nc是列表，则只取第一个元素
            if isinstance(self.yaml["nc"], list):
                assert len(self.yaml["nc"]) == 1  # 确保只有一个元素
                self.yaml["nc"] = self.yaml["nc"][0]  # 获取第一个元素

            self.names = [str(i) for i in range(self.yaml["nc"])]  # 默认的类别名称（数字）

            # 构建步幅和锚点
            m = self.model[-1]  # 获取最后一层（通常是检测层）
            if isinstance(m, Detect):  # 如果最后一层是Detect层
                s = 256  # 最小步幅的两倍
                m.inplace = self.inplace  # 设置inplace
                m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # 通过一次前向推理计算步幅
                self.stride = m.stride  # 保存步幅
                m.bias_init()  # 初始化偏置，只执行一次

                # 初始化权重和偏置
                initialize_weights(self)

        # 如果需要verbose并且LOCAL_RANK为0或-1，则输出模型信息
        if verbose and LOCAL_RANK in [0, -1]:
            self.info()
            LOGGER.info("")

    # 定义模型的前向传播方法
    def forward(self, x, profile=False, visualize=False):
        return self.forward_once(x, profile, visualize)  # 单尺度推理，用于训练

    # 单次前向传播
    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # 初始化输出和时间记录
        for m in self.model:  # 遍历所有层
            if m.f != -1:  # 如果当前层的输入层不是-1（表示不是前一层的输出）
                # 根据当前层的输入索引获取相应的输入数据
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # 如果需要记录性能，计算FLOPs并记录时间
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # 计算FLOPs
                t = time_sync()  # 获取当前时间
                for _ in range(10):  # 测试10次，计算平均时间
                    _ = m(x)  # 前向传播
                dt.append((time_sync() - t) * 100)  # 记录时间
                # 输出推理时间和FLOPs
                if m == self.model[0] and LOCAL_RANK in [0, -1]:
                    LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                if LOCAL_RANK in [0, -1]:
                    LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")

            # 执行当前层的前向传播
            x = m(x)  # 当前层的前向传播
            # 如果当前层是需要保存的层，则将输出保存
            y.append(x if m.i in self.save else None)

            # 如果需要可视化特征图，则进行可视化
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)

        # 如果需要记录性能，则输出总时间
        if profile and LOCAL_RANK in [0, -1]:
            LOGGER.info("%.1fms total" % sum(dt))

        # 如果没有头部（只包含骨干网络和其他部分），则返回所有保存的输出
        if hasattr(self, "without_head") and self.without_head:
            return y

        return x  # 返回最终的输出

    # 打印偏置（仅用于检测模块）
    def _print_biases(self):
        m = self.model[-1]  # 获取最后一层
        if not isinstance(m, Detect):  # 如果最后一层不是检测模块，直接返回
            return
        for mi in m.m:  # 遍历检测模块中的所有子模块
            b = mi.bias.detach().view(m.na, -1).T  # 将偏置转换为合适的形状
            LOGGER.info(
                ("%6g Conv2d.bias:" + "%10.3g" * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean())
            )  # 打印偏置的统计信息

    # 融合卷积层和BatchNorm层
    def fuse(self):  # 融合Conv2d()和BatchNorm2d()层
        if LOCAL_RANK in [0, -1]:
            LOGGER.info("Fusing layers... ")
        for m in self.model.modules():  # 遍历所有模块
            if type(m) is Conv and hasattr(m, "bn"):  # 如果模块是卷积并且有BatchNorm层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 融合卷积和BatchNorm
                delattr(m, "bn")  # 删除BatchNorm层
                m.forward = m.fuseforward  # 更新forward方法
        self.info()  # 输出模型信息
        return self  # 返回当前模型

    # 打印模型的信息
    def info(self, verbose=False, img_size=640):
        prefix = "Model Summary:"  # 默认前缀为“Model Summary”
        if hasattr(self, "without_head") and self.without_head:
            # 如果没有头部，则输出“Backbone summary”信息
            prefix = "Backbone summary:"
        model_info(self, verbose, img_size, prefix=prefix)  # 输出模型摘要

# 解析模型结构的函数
def parse_model(yaml_config, ch, without_head=False, verbose=True):
    if verbose and LOCAL_RANK in [0, -1]:
        LOGGER.info("\n%3s%18s%3s%10s  %-40s%-30s" % ("", "from", "n", "params", "module", "arguments"))
    gd, gw = yaml_config["depth_multiple"], yaml_config["width_multiple"]  # 深度和宽度的增益因子
    max_channels = yaml_config.get("max_channels", 1024)  # 获取最大通道数，默认1024

    nc = yaml_config["nc"]  # 获取类别数

    layers, save, c2 = [], [], ch[-1]  # 初始化层、保存层和输出通道数

    # 根据是否包含头部选择骨干网络、颈部和头部
    if without_head:
        module_args = yaml_config["backbone"]  # 如果没有头部，只有骨干网络
    else:
        module_args = yaml_config["backbone"]
        if yaml_config.get("neck"):
            module_args += yaml_config["neck"]  # 如果有neck（颈部），加上neck
        module_args += yaml_config["head"]  # 添加头部（检测模块）

    # 遍历模块参数，逐层构建模型
    for i, (f, n, m, args) in enumerate(module_args):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # 如果模块是字符串类型，进行评估
        args, _, n, c2, m_ = get_next_layer_from_cfg(gd, ch, gw, nc, m, n, f, args, max_channels)
        t = str(m)[8:-2].replace("__main__.", "")  # 获取模块的类型
        np = sum([x.numel() for x in m_.parameters()])  # 计算模块参数的数量
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # 记录模块的索引、输入层索引、类型和参数数量

        # 打印模块信息
        if verbose and LOCAL_RANK in [0, -1]:
            LOGGER.info("%3s%18s%3s%10.0f  %-40s%-30s" % (i, f, n, np, t, args))

        layers.append(m_)  # 添加模块到模型中
        if i == 0:
            ch = []  # 初始化通道列表
        ch.append(c2)  # 将当前层的输出通道数添加到通道列表
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # 将当前层的输出添加到保存列表中

        # 打印模型的结构信息
        if verbose and LOCAL_RANK in [0, -1]:
            logger.info("model build info: ", c2, ch)

    # 如果没有头部，处理neck和head部分
    if without_head:
        i = len(layers)
        for h_layer in yaml_config["neck"] + yaml_config["head"]:  # 遍历neck和head部分
            f = h_layer[0]
            f = [f] if isinstance(f, int) else f
            save.extend(x % i for x in f if x != -1 and x < len(layers))  # 将输出添加到保存列表中
            i += 1

    return nn.Sequential(*layers), sorted(save), ch  # 返回构建的模型、保存的层和输出通道数

# 根据配置生成下一层的通道数和模块
def get_next_layer_from_cfg(gd, ch, gw, nc, m, n, f, args, max_channels):
    for j, a in enumerate(args):
        try:
            args[j] = eval(a) if isinstance(a, str) else a  # 如果是字符串，进行评估
        except (ValueError, SyntaxError, NameError, TypeError):
            pass  # 如果评估出错，则跳过

    c2 = None
    n = n_ = max(round(n * gd), 1) if n > 1 else n  # 计算深度增益
    # 处理各种模块类型的输入和输出
    if m in [
        Conv,
        GhostConv,
        Bottleneck,
        GhostBottleneck,
        SPP,
        SPPF,
        DWConv,
        MixConv2d,
        Focus,
        CrossConv,
        BottleneckCSP,
        C3,
        C3TR,
        C3SPP,
        C2f,
        C2,
    ]:
        c1, c2 = ch[f], args[0]  # 获取输入通道和输出通道
        if all([c2 != nc_ for nc_ in nc]):  # 如果输出不是类别数
            c2 = make_divisible(min(c2, max_channels) * gw, 8)  # 限制输出通道数

        args = [c1, c2, *args[1:]]  # 更新参数
        # 对特定模块增加重复次数
        if m in [BottleneckCSP, C3, C3TR, C2f, C2]:
            args.insert(2, n)  # 添加重复次数
            n = 1
    # 其他模块类型的处理
    elif m is nn.BatchNorm2d:
        args = [ch[f]]  # BatchNorm的输入通道
    elif m is Concat:
        c2 = sum([ch[x] for x in f])  # Concat层的输出通道是输入通道的和
    elif m in [Detect]:
        if len(args) == 0:
            nc_ = nc.pop(0)  # 获取类别数
            args.append(nc_)
        elif isinstance(args[0], list):
            args[0] = args[0][0]

        args.append([ch[x] for x in f])  # 追加检测层的输入通道
        c2 = None  # 检测层不需要明确的输出通道
    elif m is Contract:
        c2 = ch[f] * args[0] ** 2  # Contract层的输出通道数
    elif m is Expand:
        c2 = ch[f] // args[0] ** 2  # Expand层的输出通道数
    else:
        c2 = ch[f]  # 其他层的输出通道数

    # 返回更新后的参数和构建的模块
    module = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
    return args, nc, n_, c2, module  # 返回新层的参数、类别数、重复次数、输出通道和模块
