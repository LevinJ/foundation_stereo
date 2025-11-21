import warnings, argparse, logging, os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
os.environ["XFORMERS_DISABLED"] = "1"
import omegaconf, yaml, torch,pdb
from omegaconf import OmegaConf
from core.foundation_stereo import FoundationStereo


class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        with torch.amp.autocast('cuda', enabled=True):
            disp = FoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True)
        return disp



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=f'{code_dir}/../output/foundation_small_288_960_disp64.onnx', help='Path to save results.')
    parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
    args = parser.parse_args()

    # args.height = (args.height // 3) 
    # args.width = (args.width // 3) 
    # args.valid_iters = 4
    # logging.info(f'Input image size: {args.height}x{args.width}')

    # args.valid_iters = 6

    args.height = 192
    args.width = 640
    args.valid_iters = 4
    args.save_path = f'{code_dir}/../output/foundation_small_{args.height}_{args.width}_disp64_{args.valid_iters}.onnx'
    args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug_0.5/checkpoint_epoch_500.pth'
    
    # cfg_file = '/media/levin/DATA/checkpoints/foundationstereo/23-51-11/cfg.yaml'
    cfg_file = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/cfg.yaml'
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(cfg_file)
    for k in args.__dict__:
      cfg[k] = args.__dict__[k]
    if 'vit_size' not in cfg:
      cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    global_step = ckpt.get('global_step', 'N/A')
    logging.info(
        f"ckpt global_step:{global_step}, epoch:{ckpt['epoch']}")
    model_state = None
    if 'model' in ckpt:
        model_state = ckpt['model']
    else:
        model_state = ckpt['model_state']
    model.load_state_dict(model_state)
    model.cuda()
    model.eval()


    left_img = torch.randn(1, 3, args.height, args.width).cuda().float()
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float()

    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        opset_version=16,
        input_names = ['left', 'right'],
        output_names = ['disp'],
        dynamic_axes={
            'left': {0 : 'batch_size'},
            'right': {0 : 'batch_size'},
            'disp': {0 : 'batch_size'}
        },
        verbose=True,
    )

