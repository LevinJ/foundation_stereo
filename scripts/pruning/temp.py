import torch
import torch.nn as nn
import torch_pruning as tp

class DynamicLoopCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)   # 8 通道输出

    def forward(self, x):
        feat = self.conv(x)          # (B, 8, H, W)
        # ===== 动态 for-loop 拼 list =====
        tmp = []
        n = feat.size(2) // 10
        for i in range(n):           # 循环长度是 Python int，tracer 无法静态展开
            tmp.append(feat[:, i:i+1, ...])   # 每次取 1 通道
        # cat 沿着通道拼 → 期望 3 通道
        out = torch.cat(tmp, dim=1)  # 这里输入通道数 = [1, 1, 1]
        out = 2 * out
        return out                   # (B, 3, H, W)

model = DynamicLoopCat()
DG = tp.DependencyGraph()

# 下面一行会触发 TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'
DG.build_dependency(model, example_inputs=torch.randn(1, 3, 32, 32))

group = DG.get_pruning_group(model.conv, tp.prune_conv_out_channels, idxs=[2,6,9])

# 3. 打印依赖关系
print(group.details())