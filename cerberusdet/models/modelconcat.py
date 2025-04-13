import torch
import torch.nn as nn
from ultralytics import YOLO
from copy import deepcopy

class MultitaskModel(nn.Module):
    def __init__(self, layer_list):
        """
        读取模型
        layer_list: 带读取的的模块列表，列表内容为 tuple(name, module)
                    name为 str ，module为通常module
        """
        super().__init__()
        self.model = nn.ModuleList([module for _, module in layer_list])

        # 模型中，Concat需要特殊输入，其存在于层数为索引大于9，且 
        # 索引%13 in [11, 1, 4, 7]，分别对应需要的层数在此做记录，
        # 以便在forward方法中特殊处理
        self.concat_dict = {11: 6, 
                            1: 4, 
                            4: 12, 
                            7: 9}
        # 模型中，由于YOLO Detect与Concat块，需要特殊层索引的输出
        self.output_note = [2, 5, 12]


        self.output_cache = {}  # 缓存中间层输出

    def forward(self, x):
        self.output_cache = {}
        res = {}
        taskid = 1
        for idx, layer in enumerate(self.model):
            # 判断当前是否为Concat
            if idx > 9 and idx % 13 in self.concat_dict:
                extrainputidx = self.concat_dict[idx % 13]
                
                # 特殊处理
                if extrainputidx == 12:
                    extrainputidx = idx - 5
                
                extrax = self.output_cache[extrainputidx]
                x = torch.cat([x, extrax], dim=1)
                # 此时已经执行完Concat操作
                continue

            # 判断是否为Detect
            elif idx > 9 and idx % 13 == 9:
                extra_input_ids = [idx-7, idx-4]
                y = []
                for id in extra_input_ids:
                    y.append(self.output_cache[id])
                y.append(deepcopy(x))
                x = y
                x = layer(x)

                # 记录输出，并提高taskid
                res[f"task{taskid}"] = deepcopy(x)
                taskid += 1
                continue
    
            # 判断是否为第一个Upsample
            elif idx % 13 == 10:
                x = self.output_cache[9]
            
            x = layer(x)

            # 判断是否需要记录输出
            if idx in [4, 6, 9] or (idx % 13 in self.output_note and idx > 9):
                self.output_cache[idx] = deepcopy(x)


        return res

# 示例运行
if __name__ == "__main__":

    taskid = "aaa"
    pt_file = "pretrained/client_model.pt"

    # 加载 YOLO 模型
    model = YOLO(pt_file)

    # print(yolomodel)
    model = model.model # 这里能够正常输入输出
    input_tensor = torch.randn(1, 3, 640, 640)
    outputs = model(input_tensor)
    print(outputs["task1"][0].shape)
    print(outputs["task1"][1][0].shape)
    print(outputs["task1"][1][1].shape)
    print(outputs["task1"][1][2].shape)

    # print(models)
    model = model.blocks
    # print(model)

    # 获取 backbone 层（此处 为第 1 层）
    backbone_layers = list(model.children())[0].model
    # print(backbone_layers)
    
    
    # 拆分前两层
    layers = []

    for idx, (name, module) in enumerate(backbone_layers.named_children()): 
    # 此处写named_module会拆到最底层，而named_children只会拆浅层

        layers.append((name, deepcopy(module)))
  
    for idx, (name, module) in enumerate(list(model.named_children())[1:]):
        
        layers.append((name, deepcopy(module)))
    
    # 尝试两分支
    for idx, (name, module) in enumerate(list(model.named_children())[1:]):
        
        layers.append((name, deepcopy(module)))

    print(f"拆分出的层：{[name for name, _ in layers]}")

    multitask = MultitaskModel(layers)
    input_tensor = torch.randn(1, 3, 640, 640)
    outputs = multitask(input_tensor)
    print(outputs)
    # print(outputs[0].shape)
    # print(outputs[1][0].shape)
    # print(outputs[1][1].shape)
    # print(outputs[1][2].shape)
    # print(multitask)



