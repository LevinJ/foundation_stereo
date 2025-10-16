# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time  # Add this import at the top of the file
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch.distributed as dist
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')

from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
from torchvision import transforms
from scripts.distillation.evaluation.metric_per_image import epe_metric, d1_metric, threshold_metric,bpx_metric
from scripts.distillation.utils import common_utils
from torch.utils.tensorboard import SummaryWriter
from scripts.distillation.utils.common_utils import color_map_tensorboard, write_tensorboard

class TransposeImage(object):
    def __init__(self):
        return

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample
class ToTensor(object):
    def __init__(self):
        return

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                if k == 'super_pixel_label':
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.int32)
                elif k in ['occ_mask', 'occ_mask_2']:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.bool)
                else:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample
class DivisiblePad(object):
    def __init__(self, divis_by=32, mode='round'):
        self.by = divis_by
        self.mode = mode

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_h = self.by - h % self.by
        else:
            pad_h = 0
        if w % self.by != 0:
            pad_w = self.by - w % self.by
        else:
            pad_w = 0

        if self.mode == 'round':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        elif self.mode == 'tr':
            pad_top = pad_h
            pad_right = pad_w
            pad_bottom = 0
            pad_left = 0
        else:
            raise Exception('no DivisiblePad mode')

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = [pad_top, pad_right, pad_bottom, pad_left]
        return sample
class ZedDataset(Dataset):
    def __init__(self, config_txt, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        with open(config_txt, 'r') as f:
            for line in f:
                left, right, disp = line.strip().split()
                self.samples.append({
                    'left': os.path.join(root_dir, left),
                    'right': os.path.join(root_dir, right),
                    'disp': os.path.join(root_dir, disp)
                })
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        left_img = np.array(Image.open(sample_info['left']).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(sample_info['right']).convert('RGB'), dtype=np.float32)

        # disp_img = np.load(sample_info['disp']).astype(np.float32) 
        # Load the disparity images
        if sample_info['disp'].endswith('.npy'):
            disp_img = np.load(sample_info['disp']).astype(np.float32)
        else:
            disp_img = np.array(Image.open(sample_info['disp']), dtype=np.float32) / 256.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        if self.transform:
            sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = sample_info['left']
        return sample



class FoundationStereoEvaluator:
    def __init__(self, config_txt, root_dir, args):
        self.setup_logging(args)
        logger = self.logger
        set_seed(0)
        torch.autograd.set_grad_enabled(False)

        self.args = args
        ckpt_dir = args.ckpt_dir
        valid_iters = args.valid_iters
        cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        # cfg['ckpt_dir'] = ckpt_dir
        # cfg['valid_iters'] = valid_iters
        self.args = OmegaConf.create(cfg)
        # logger.info(f"args:\n{self.args}")
        logger.info(f"Using pretrained model from {ckpt_dir}")

        

        self.model = FoundationStereo(self.args)
        ckpt = torch.load(ckpt_dir)
        logger.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()
        self.model.eval()

        # Define Transformations
        data_transforms = transforms.Compose([
            DivisiblePad(divis_by=32, mode='round'),
            TransposeImage(),
            ToTensor(),
        ])
        self.dataset = ZedDataset(config_txt, root_dir, transform=data_transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.valid_iters = valid_iters
        return
    def setup_logging(self, args):
        args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, 'eval'))
        os.makedirs(args.output_dir, exist_ok=True)
        if args.dist_mode:
            dist.barrier()
        # log
        if args.dist_mode:
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = int(os.environ["RANK"])
        else:
            local_rank = 0
            global_rank = 0
        # log_file = os.path.join(args.output_dir, 'eval_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        log_file = None
        logger = common_utils.create_logger(log_file, rank=local_rank)
        tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'eval_tensorboard')) if global_rank == 0 else None

        self.logger = logger
        self.tb_writer = tb_writer
        self.local_rank = local_rank
        self.global_rank = global_rank
        return

    def cal_metrics(self, disp_pred, disp_gt, metric_func_dict, epoch_metrics, data, i, infer_time):
        local_rank = self.local_rank
        current_epoch = 0
        mask = (disp_gt > 0)

        for m in metric_func_dict:
            metric_func = metric_func_dict[m]
            res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
            epoch_metrics[m]['indexes'].extend(data['index'].tolist())
            epoch_metrics[m]['values'].extend(res.tolist())

        if i % 10 == 0:
            message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                        ).format(current_epoch, i, len(self.dataloader), infer_time * 1000)
            self.logger.info(message)

            if self.tb_writer is not None:
                img = torch.cat([data['left'][0], data['right'][0]], dim=1)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                tb_info = {
                    'image/eval/image': img,
                    'image/eval/disp': color_map_tensorboard(data['disp'][0], disp_pred.squeeze(1)[0])
                }
                write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.dataloader) + i)

        if i == len(self.dataloader) - 1:
            self.logger.info(f"Finish evaluation epoch {current_epoch}, start to gather metrics.")
            # gather from all gpus
            if self.args.dist_mode:
                dist.barrier()
                self.logger.info("Start reduce metrics.")
                for k in epoch_metrics.keys():
                    indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
                    values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
                    gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                    gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_indexes, indexes)
                    dist.all_gather(gathered_values, values)
                    unique_dict = {}
                    for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(),
                                        torch.cat(gathered_values, dim=0).tolist()):
                        if key not in unique_dict:
                            unique_dict[key] = value
                    epoch_metrics[k]["indexes"] = list(unique_dict.keys())
                    epoch_metrics[k]["values"] = list(unique_dict.values())

            results = {}
            for k in epoch_metrics.keys():
                results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

            if local_rank == 0 and self.tb_writer is not None:
                tb_info = {}
                for k, v in results.items():
                    tb_info[f'scalar/val/{k}'] = v.item()

                write_tensorboard(self.tb_writer, tb_info, current_epoch)

            self.logger.info(f"Epoch {current_epoch} metrics: {results}")
        return

    def evaluate(self):
        local_rank = self.local_rank
        # total_epe = 0.0
        # count = 0
        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'bp_1': partial(bpx_metric, x=1),
            'bp_2': partial(bpx_metric, x=2),
            # 'thres_1': partial(threshold_metric, threshold=1),
            # 'thres_2': partial(threshold_metric, threshold=2),
            # 'thres_3': partial(threshold_metric, threshold=3),
        }
        epoch_metrics = {}
        for k in metric_func_dict.keys():
            epoch_metrics[k] = {'indexes': [], 'values': []}
        # for sample in self.dataloader:
        for i, data in enumerate(self.dataloader):
            for k, v in data.items():
                data[k] = v.to(local_rank) if torch.is_tensor(v) else v
            # Model inference
            with torch.no_grad():
                infer_start = time.time()
                pred_disp = self.model.forward(data['left'], data['right'], iters=self.valid_iters, test_mode=True)
                infer_time = time.time() - infer_start
            self.cal_metrics(pred_disp, data['disp'], metric_func_dict, epoch_metrics, data, i, infer_time)
            # Compute metrics
            # mask = (disp_gt > 0)
            # epe = torch.mean(torch.abs(pred_disp.squeeze(1) - disp_gt)[mask]).item()
            # total_epe += epe
            # count += 1
            # print(f"Sample {count}: EPE = {epe}")
        # mean_epe = total_epe / count if count > 0 else 0.0
        # print(f"Mean EPE over dataset: {mean_epe}")

if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir', default=f'{code_dir}/../../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--exp_group_path', type=str, default='zed_fs', help='experiment group path')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='enable distributed mode')

    args = parser.parse_args()
    
    # config_txt = '/home/levin/workspace/temp/OpenStereo/data/ZED/zed_250601.txt'
    # root_dir = '/media/levin/DATA/nerf/new_es8/stereo/'

    #for kitti12
    config_txt = '/home/levin/workspace/temp/OpenStereo/data/KITTI12/kitti12_train180_0.txt'
    root_dir = '/media/levin/DATA/nerf/public_depth/kitti12/'
    evaluator = FoundationStereoEvaluator(config_txt, root_dir, args)
    evaluator.evaluate()
