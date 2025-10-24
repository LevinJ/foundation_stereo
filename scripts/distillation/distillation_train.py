# @Time    : 2023/8/28 22:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import tqdm
import shutil
import torch
import torch.distributed as dist

from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT

import warnings
import copy
# 同时忽略 FutureWarning 和 UserWarning （包括PyTorch产生的）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from Utils import freeze_model

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # mode
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, required=False, help='specify the config for training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')

    # save path
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--cover_old_exp', action='store_true', default=False)

    # dataloader
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')

    args = parser.parse_args()
    args.dist_mode = False 
    args.extra_tag = 'debug'
    args.cfg_file = 'cfgs/foundationstereo/fstereo_sceneflow.yaml'

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    dataset_names = [x.DATASET for x in cfgs.DATA_CONFIG.DATA_INFOS]
    unique_dataset_names = list(set(dataset_names))
    if len(unique_dataset_names) == 1:
        exp_dataset_dir = unique_dataset_names[0]
    else:
        exp_dataset_dir = 'MultiDataset'
    args.exp_group_path = os.path.join(exp_dataset_dir, cfgs.MODEL.NAME)
    args.tag = os.path.basename(args.cfg_file)[:-5]

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]
        assert os.path.exists(each.DATA_PATH), '[Errno 2] No such file or directory: {}, You must modify the data root path in cfgs/databasic.py to the path of your own dataset.'.format(each.DATA_PATH)

    args.run_mode = 'train'
    return args, cfgs



class DistillationTrainer:
    def __init__(self, args, cfgs):
        self.args = args
        self.cfgs = cfgs
        self.local_rank = 0
        self.global_rank = 0
        self.group_rank = 0
        self.logger = None
        self.tb_writer = None
        self.model_trainer = None
        self.tbar = None
        self._setup_env()
        self._setup_dirs()
        self._setup_logger()
        self._setup_trainer()

    def _setup_env(self):
        if self.args.dist_mode:
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.group_rank = int(os.environ["GROUP_RANK"])
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.group_rank = 0
        torch.cuda.set_device(self.local_rank)
        if self.args.fix_random_seed:
            seed = 0 if not self.args.dist_mode else dist.get_rank()
            common_utils.set_random_seed(seed=seed)

    def _setup_dirs(self):
        self.args.output_dir = str(os.path.join(self.args.save_root_dir, self.args.exp_group_path, self.args.tag, self.args.extra_tag))
        if os.path.exists(self.args.output_dir) and self.args.cover_old_exp and self.global_rank == 0:
            shutil.rmtree(self.args.output_dir)
        if self.args.dist_mode:
            dist.barrier()
        if os.path.exists(self.args.output_dir) and self.args.extra_tag != 'debug' and self.cfgs.MODEL.CKPT == -1:
            raise Exception('There is already an exp with this name')
        self.args.ckpt_dir = os.path.join(self.args.output_dir, 'ckpt')
        if not os.path.exists(self.args.ckpt_dir) and self.local_rank == 0:
            os.makedirs(self.args.ckpt_dir, exist_ok=True)
        if self.args.dist_mode:
            dist.barrier()

    def _setup_logger(self):
        log_file = os.path.join(self.args.output_dir, 'train_{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), self.group_rank))
        self.logger = common_utils.create_logger(log_file, rank=self.local_rank)
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.args.output_dir, 'tensorboard')) if self.global_rank == 0 else None
        for key, val in vars(self.args).items():
            self.logger.info('{:16} {}'.format(key, val))
        common_utils.log_configs(self.cfgs, logger=self.logger)
        if self.global_rank == 0:
            os.system('cp %s %s' % (self.args.cfg_file, self.args.output_dir))

    def _setup_trainer(self):
        teacher_cfgs = copy.deepcopy(self.cfgs)
        teacher_args = copy.deepcopy(self.args)
        teacher_args.run_mode = 'train_teacher'
        self.model_trainer_teacher = build_trainer(teacher_args, teacher_cfgs, self.local_rank, self.global_rank, self.logger, self.tb_writer)
        self.model_trainer_teacher.model = freeze_model(self.model_trainer_teacher.model)

        #modify student cfgs
        self.cfgs.MODEL.PRETRAINED_MODEL = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/model_best_bp2.pth'
        self.cfgs.MODEL.vit_size = 'vits'
        self.cfgs.MODEL.train_iters = 0
        self.cfgs.MODEL.valid_iters = 0
        self.cfgs.trainer_teacher = self.model_trainer_teacher
        self.model_trainer = build_trainer(self.args, self.cfgs, self.local_rank, self.global_rank, self.logger, self.tb_writer)
        self.tbar = tqdm.trange(self.model_trainer.last_epoch + 1, self.model_trainer.total_epochs,
                               desc='epochs', dynamic_ncols=True, disable=(self.local_rank != 0),
                               bar_format='{l_bar}{bar}{r_bar}\n')

    def run(self):
        for current_epoch in self.tbar:
            self.model_trainer.train(current_epoch, self.tbar)
            self.model_trainer.save_ckpt(current_epoch)
            if current_epoch % self.cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == self.model_trainer.total_epochs - 1:
                self.model_trainer.evaluate(current_epoch)


if __name__ == '__main__':
    args, cfgs = parse_config()
    trainer = DistillationTrainer(args, cfgs)
    trainer.run()
