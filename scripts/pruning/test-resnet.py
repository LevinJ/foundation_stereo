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

class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_layer = nn.Linear(10, 10)

    def forward(self, x):
        # Create a new layer only if it doesn't exist
        if not hasattr(self, 'dynamic_layer'):
            self.dynamic_layer = nn.Linear(10, 10)
        return self.dynamic_layer(self.static_layer(x))

model = DynamicModel()
DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 10, 10))

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