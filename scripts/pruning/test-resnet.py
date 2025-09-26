import torch
from torchvision.models import resnet18
import torch_pruning as tp
import torch.nn as nn

# class DynamicCat(nn.Module):
#     def forward(self, x):
#         a = x                      # (N,3,H,W)
#         tmp = []
#         for i in range(3):         # 动态 list
#             tmp.append(a[:, i:i+1, ...])
#         out = torch.cat(tmp, dim=1)   # tracer 无法静态展开循环
#         return out

# model = DynamicCat()
# DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))

import torch
import torch.nn as nn
import torch_pruning as tp

class SkipConnectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv_skip = nn.Conv2d(3, 16, 1) # 1x1 conv for the skip connection
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # The skip connection: 
        # x_skip is created from the input x
        x_skip = self.conv_skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # The skip connection is added here
        # x = x + x_skip 
        x = torch.cat([x, x_skip], dim=1)
        x = self.relu(x)
        return x

model = SkipConnectionModel()
example_inputs = torch.randn(1, 3, 32, 32)

DG = tp.DependencyGraph().build_dependency(model, example_inputs=(example_inputs,))

group = DG.get_pruning_group(model.conv1, tp.prune_conv_out_channels, idxs=[2,6,9])

# 3. 打印依赖关系
print(group.details())


example_inputs = torch.randn(1,3,224,224)

model = resnet18(pretrained=True).eval()

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )




base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
tp.utils.print_tool.before_pruning(model) # or print(model)

# 3. prune all grouped layers that are coupled with model.conv1 (included).
if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
    group.prune()

tp.utils.print_tool.after_pruning(model) # or print(model), this util will show the difference before and after pruning
macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")