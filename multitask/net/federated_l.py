import torch


def federated_pt(pt1, pt2):
    """
    使用复数个pt模型,共享参数的根部部分。
    """
# 后续可以换成*pt

    model1_state = torch.load("model1.pt")['model'].state_dict()
    model2_state = torch.load("model2.pt")['model'].state_dict()

    backbone_keys = [k for k in model1_state.keys() if k.startswith("model.0.") or ...]  # 调整范围
    averaged_backbone = {k: (model1_state[k] + model2_state[k]) / 2 for k in backbone_keys}

    new_model1_state = {k: averaged_backbone[k] if k in backbone_keys else model1_state[k] for k in model1_state.keys()}
    new_model2_state = {k: averaged_backbone[k] if k in backbone_keys else model2_state[k] for k in model2_state.keys()}

    # 这里应该选择YOLO的save方法
    torch.save({'model': new_model1_state}, "new_model1.pt")
    torch.save({'model': new_model2_state}, "new_model2.pt")

    pass

