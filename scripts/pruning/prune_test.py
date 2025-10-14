

from pyexpat import model
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch_pruning as tp

from omegaconf import OmegaConf
import logging
import os,sys

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
from core.foundation_stereo import FoundationStereo

class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        with torch.amp.autocast('cuda', enabled=True):
            disp = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
        return disp

class ResNetPruner:
    def print_named_modules(self, model, only_leaf=True):
        print("\nAll modules in model.named_modules():")
        total_params = sum(p.numel() for p in model.parameters())
        total_mb = total_params * 4 / (1024 ** 2)
        total_billion = total_params / 1e9
        print(f"Total parameters: {total_params} ({total_mb:.2f} MB, {total_billion:.4f} Billion)")
        idx = 0
        for name, module in model.named_modules():
            if only_leaf:
                # A leaf module has no children
                if len(list(module.children())) == 0:
                    print(f"{idx}: {name}: {module.__class__.__name__} | {module}")
                    idx += 1
            else:
                print(f"{idx}: {name}: {module.__class__.__name__} | {module}")
                idx += 1
        return
    def __init__(self):
        return
    def get_fs_model(self):
        # Load FoundationStereoOnnx model as self.model2 (GPU, with ONNX-style forward)
        code_dir = os.path.dirname(os.path.realpath(__file__))
        ckpt_dir = f'{code_dir}/../../pretrained_models/11-33-40/model_best_bp2.pth'
        cfg_path = f'{os.path.dirname(ckpt_dir)}/cfg.yaml'
        cfg = OmegaConf.load(cfg_path)
        # Simulate argparse.Namespace as in make_onnx.py
        class Args:
            pass
        args = Args()
        args.save_path = f'{code_dir}/../../output/foundation_small_288_960_disp64.onnx'
        args.ckpt_dir = ckpt_dir
        args.height = 288
        args.width = 960
        args.valid_iters = 16
        # Merge args into cfg as in make_onnx.py
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        args = OmegaConf.create(cfg)
        self.model2 = FoundationStereoOnnx(args)
        ckpt = torch.load(ckpt_dir, weights_only=False)
        self.model2.load_state_dict(ckpt['model'], strict=False)
        self.model2.cuda()
        self.model2.eval()
        self.args = args
        return self.model2, self.args
    
    def prune_fs2(self, model, args):
        imag1 = torch.randn(1, 3, args.height, args.width).cuda().float()
        imag2 = torch.randn(1, 3, args.height, args.width).cuda().float()
        example_inputs = (imag1, imag2)
        
        ignored_layers = []
        # for name, m in model.named_modules():
        #     if name.startswith('update_block') or name.startswith('feature'):
        #         ignored_layers.append(m)  
                
        DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs, ignored_layers=ignored_layers)

        # 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
        group = DG.get_pruning_group(model.update_block.encoder.convc1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )


        # 3. Prune the model
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        tp.utils.print_tool.before_pruning(model) # or print(model)
        if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
            group.prune()
            print(group)
        tp.utils.print_tool.after_pruning(model) # or print(model), this util will show the difference before and after pruning
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
        return
    def prune_fs(self, model, args):
        imag1 = torch.randn(1, 3, args.height, args.width).cuda().float()
        imag2 = torch.randn(1, 3, args.height, args.width).cuda().float()
        example_inputs = (imag1, imag2)
         # 1. Importance criterion, here we calculate the L2 Norm of grouped weights as the importance score
        imp = tp.importance.GroupMagnitudeImportance(p=2) 

        # 2. Initialize a pruner with the model and the importance criterion
        ignored_layers = []
        for name, m in model.named_modules():
            # DO NOT prune the final classifier!
            if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
                ignored_layers.append(m)
            # DO NOT prune the upsampling weight layer (must have 9 output channels for context_upsample)
            # if isinstance(m, torch.nn.Conv2d) and m.out_channels == 9:
            #     ignored_layers.append(m)
            # DO NOT prune modules named 'spx_gru'
            if name.endswith('spx_gru'):
                ignored_layers.append(m)
            # DO NOT prune the upsampling weight layer (must have 9 output channels for context_upsample)
            # if isinstance(m, torch.nn.Conv2d) and m.out_channels == 9:
            #     ignored_layers.append(m)

        pruner = tp.pruner.BasePruner( # We can always choose BasePruner if sparse training is not required.
            model,
            example_inputs,
            importance=imp,
            pruning_ratio=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            # pruning_ratio_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized pruning ratios for layers or blocks
            ignored_layers=ignored_layers,
            # round_to=8, # It's recommended to round dims/channels to 4x or 8x for acceleration. Please see: https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
        )

        # 3. Prune the model
        base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        tp.utils.print_tool.before_pruning(model) # or print(model)
        pruner.step()
        tp.utils.print_tool.after_pruning(model) # or print(model), this util will show the difference before and after pruning
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")
        return

    def run(self):
        """
        Run all pruning steps with hardcoded parameters.
        """
        # return self.test_resnet18()
        model, args = self.get_fs_model()
        self.print_named_modules(model)
        self.prune_fs2(model, args)

        # 1. Build dependency graph for a resnet18. This requires a dummy input for forwarding
        # imag1 = torch.randn(1, 3, args.height, args.width).cuda().float()
        # imag2 = torch.randn(1, 3, args.height, args.width).cuda().float()

        # DG = tp.DependencyGraph().build_dependency(model, example_inputs=(imag1, imag2))
        # # DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=torch.randn(1,3,224,224))

        # for group in DG.get_all_groups(ignored_layers=[], root_module_types=[nn.Conv2d, nn.Linear]):
        #     # Handle groups in sequential order
        #     idxs = [2,4,6] # your pruning indices, feel free to change them
        #     group.prune(idxs=idxs)
        #     print(group)

        # # 2. To prune the output channels of model.conv1, we need to find the corresponding group with a pruning function and pruning indices.
        # group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )

        # # 3. Do the pruning
        # if DG.check_pruning_group(group): # avoid over-pruning, i.e., channels=0.
        #     group.prune()

if __name__ == "__main__":
    pruner = ResNetPruner()
    pruner.run()
