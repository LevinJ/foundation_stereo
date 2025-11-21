# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from core.foundation_stereo import *
from Utils import *
from core.utils.utils import InputPadder
from omegaconf import OmegaConf
import os
import sys
import time  # Add this import at the top of the file
from typing import Optional, Tuple
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from  scripts.zed_preprocess import ZedPreprocessor


if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--left_file', default=f'{code_dir}/../assets/left.png', type=str)
    parser.add_argument(
        '--right_file', default=f'{code_dir}/../assets/right.png', type=str)
    parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt',
                        type=str, help='camera intrinsic matrix and baseline file')
    parser.add_argument(
        '--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument(
        '--out_dir', default=f'{code_dir}/../output/', type=str, help='the directory to save results')
    parser.add_argument('--scale', default=1, type=float,
                        help='downsize the image by scale, must be <=1')
    parser.add_argument('--hiera', default=0, type=int,
                        help='hierarchical inference (only needed for high-resolution images (>1K))')
    parser.add_argument('--z_far', default=10, type=float,
                        help='max depth to clip in point cloud')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    parser.add_argument('--get_pc', type=int, default=1,
                        help='save point cloud output')
    parser.add_argument('--remove_invisible', default=1, type=int,
                        help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
    parser.add_argument('--denoise_cloud', type=int, default=1,
                        help='whether to denoise the point cloud')
    parser.add_argument('--denoise_nb_points', type=int, default=30,
                        help='number of points to consider for radius outlier removal')
    parser.add_argument('--denoise_radius', type=float,
                        default=0.03, help='radius to use for outlier removal')
    args = parser.parse_args()

    # speed bump
    # folder = '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/jiuting_campus'
    # folder = '/media/levin/DATA/nerf/new_es8/stereo/250610'
    # # file_name = '20250331_111635.913_10.png'
    # file_name = '00000006.png'

    # # big hole
    # folder = '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/lidar'
    # file_name = '00000062.png'

    # args.left_file = f"{folder}/colored_l/{file_name}"
    # args.right_file = f"{folder}/colored_r/{file_name}"

    folder = '/media/levin/DATA/nerf/new_es8/stereo/20250702'
    #eval
    # file_name = '1751438147.4760577679.png'
    # file_name = '1751438168.9847328663.png'
    #train
    file_name ='1751438145.9038047791.png'
    # file_name ='1751438153.062338829.png'
    args.left_file = f"{folder}/left_images/{file_name}"
    args.right_file = f"{folder}/right_images/{file_name}"




    # args.intrinsic_file = "/media/levin/DATA/nerf/new_es8/stereo_20250331/K_Zed.txt"

    args.z_far = 80
    args.remove_invisible = True
    args.denoise_cloud = False

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # args.ckpt_dir = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/model_best_bp2.pth'
    # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug/ckpt/checkpoint_epoch_20.pth'
    # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug5/ckpt/checkpoint_epoch_999.pth'
    # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug/ckpt/checkpoint_epoch_999.pth'
    # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug_scale_0.25_8000/checkpoint_epoch_999.pth'
    args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug_0.5/checkpoint_epoch_500.pth'
    ckpt_dir = args.ckpt_dir
    # cfg_file = '/media/levin/DATA/checkpoints/foundationstereo/23-51-11/cfg.yaml'
    cfg_file = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/cfg.yaml'
    cfg = OmegaConf.load(cfg_file)
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    # logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    args.vit_size = 'vits'
    args.valid_iters = 4
    args.max_disp = 64

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
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

    code_dir = os.path.dirname(os.path.realpath(__file__))

    left_img_path = args.left_file
    right_img_path = args.right_file
    zp = ZedPreprocessor(scale=0.5)
    img0, img0_ori = zp.prepare(left_img_path)
    img1, img1_ori = zp.prepare(right_img_path)
    H, W = img0.shape[2], img0.shape[3]
    img0 = img0.cuda()
    img1 = img1.cuda()

    for i in range(1):
        start_time = time.time()  # Start timing
        with torch.cuda.amp.autocast(True):
            if not args.hiera:
                disp = model.forward(
                    img0, img1, iters=args.valid_iters, test_mode=True)
            else:
                disp = model.run_hierachical(
                    img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        end_time = time.time()  # End timing

        # Print the duration
        print(f"Running duration: {end_time - start_time:.2f} seconds")

    disp = disp.data.cpu().numpy().reshape(H, W)
    # vis = vis_disparity(disp)
    # vis = np.concatenate([img0_ori, vis], axis=1)
    # imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    # logging.info(f"Output saved to {args.out_dir}")

    # vis = np.concatenate([img0_ori, img1_ori], axis=1)
    # imageio.imwrite(f'{args.out_dir}/vis_orgin.png', vis)
    # logging.info(f"Output saved to {args.out_dir}")

    # imageio.imwrite(f'{args.out_dir}/rbg.png', img0_ori)
    # logging.info(f"Output saved to {args.out_dir}")

    if args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(
            disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = (us_right < 0) | (disp <= 0)
        disp[invalid] = 0
        # disp_dir = '/media/levin/DATA/nerf/new_es8/stereo/250610/disp_test'
        # os.makedirs(disp_dir, exist_ok=True)
        # np.save(f'{disp_dir}/{file_name[:-4]}.npy', disp)
        # logging.info(f"Disparity map saved to {disp_dir}/{file_name[:-4]}.npy")

    if args.get_pc:
        K = zp.updated_K().copy()
        baseline = zp.get_baseline()
        depth = K[0, 0]*baseline/disp
        # np.save(f'{args.out_dir}/depth_meter.npy', depth)
        import matplotlib.pyplot as plt

        # Display the RGB, disparity, and depth images side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].set_title("RGB Image")
        axes[0].imshow(img0_ori)
        axes[0].axis("off")

        im1 = axes[1].imshow(disp, cmap='plasma')
        axes[1].set_title("Disparity Image")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Disparity (pixels)")

        im2 = axes[2].imshow(depth)
        axes[2].set_title("Depth Image")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (meters)")

        plt.tight_layout()
        # plt.savefig(f'{args.out_dir}/rgb_and_depth.png')
        plt.show()
        process_and_visualize_point_cloud(
            depth, K, img0_ori, args.out_dir, args.z_far, args.denoise_cloud, args.denoise_nb_points, args.denoise_radius)
