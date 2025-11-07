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
    # # folder = '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/lidar'
    # # file_name = '00000062.png'

    # args.left_file = f"{folder}/colored_l/{file_name}"
    # args.right_file = f"{folder}/colored_r/{file_name}"

    folder = '/media/levin/DATA/nerf/new_es8/stereo/20250702'
    file_name = '1751438147.4760577679.png'
    args.left_file = f"{folder}/left_images/{file_name}"
    args.right_file = f"{folder}/right_images/{file_name}"




    args.intrinsic_file = "/media/levin/DATA/nerf/new_es8/stereo_20250331/K_Zed.txt"

    args.z_far = 80
    args.remove_invisible = True
    args.denoise_cloud = False

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # args.ckpt_dir = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/model_best_bp2.pth'
    # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug/ckpt/checkpoint_epoch_20.pth'
    args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug5/ckpt/checkpoint_epoch_999.pth'
    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    # logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")
    args.vit_size = 'vits'
    args.valid_iters = 0
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
    img0 = imageio.imread(args.left_file, pilmode="RGB")
    img1 = imageio.imread(args.right_file, pilmode="RGB")
    img0 = img0[:1000,...]
    img1 = img1[:1000,...]
    scale = args.scale
    assert scale <= 1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    img1_ori = img1.copy()
    logging.info(f"img0: {img0.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

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

    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    logging.info(f"Output saved to {args.out_dir}")

    vis = np.concatenate([img0_ori, img1_ori], axis=1)
    imageio.imwrite(f'{args.out_dir}/vis_orgin.png', vis)
    logging.info(f"Output saved to {args.out_dir}")

    imageio.imwrite(f'{args.out_dir}/rbg.png', img0_ori)
    logging.info(f"Output saved to {args.out_dir}")

    if args.remove_invisible:
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(
            disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = (us_right < 0) | (disp <= 0)
        disp[invalid] = 0
        disp_dir = '/media/levin/DATA/nerf/new_es8/stereo/250610/disp_test'
        os.makedirs(disp_dir, exist_ok=True)
        np.save(f'{disp_dir}/{file_name[:-4]}.npy', disp)
        logging.info(f"Disparity map saved to {disp_dir}/{file_name[:-4]}.npy")

    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(
                np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= scale
        depth = K[0, 0]*baseline/disp
        np.save(f'{args.out_dir}/depth_meter.npy', depth)
        import matplotlib.pyplot as plt

        # Display the RGB image and depth image side by side
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title("RGB Image")
        plt.imshow(img0_ori)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Depth Image")
        plt.imshow(depth)
        plt.colorbar(label="Depth (meters)")
        plt.axis("off")

        plt.tight_layout()
        # plt.savefig(f'{args.out_dir}/rgb_and_depth.png')
        plt.show()
        process_and_visualize_point_cloud(
            depth, K, img0_ori, args.out_dir, args.z_far, args.denoise_cloud, args.denoise_nb_points, args.denoise_radius)
