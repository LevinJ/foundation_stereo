import torch
import torch.nn as nn
import torch_pruning as tp

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

class CustomCatModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # self.conv2 = nn.Conv2d(16, 16, 3, padding=1)

    def forward(self, x, x2):
        res = build_concat_volume(x, x2, maxdisp=4)
        return res

# Create model and inputs
model = CustomCatModel()
example_inputs = (torch.randn(1, 3, 8, 8),torch.randn(1, 3, 8, 8))

# Try to build the dependency graph
try:
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    print("Dependency graph built successfully.")
except Exception as e:
    print(f"Failed to build dependency graph: {e}")