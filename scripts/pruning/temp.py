import torch
import torch.nn as nn
import torch_pruning as tp

class DynamicLoopCat(nn.Module):
    def __init__(self):
        super().__init__()
        volume_dim = 28
        self.corr_stem = nn.Sequential(
        nn.Conv3d(32, volume_dim, kernel_size=1),
        # BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
        # ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
        # ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
        )

        # self.corr_stem = nn.Conv3d(32, volume_dim, kernel_size=1)
        

    def forward(self, x):
        out = self.corr_stem(x)
        return out                   

model = DynamicLoopCat()
DG = tp.DependencyGraph()

# 下面一行会触发 TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
DG.build_dependency(model, example_inputs=torch.randn(1, 32, 16, 72, 240))

group = DG.get_pruning_group(model.corr_stem[0], tp.prune_conv_out_channels, idxs=[2,6,9])

# 3. 打印依赖关系
print(group.details())