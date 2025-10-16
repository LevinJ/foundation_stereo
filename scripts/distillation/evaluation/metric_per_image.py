import torch
def bpx_metric(disp_pred, disp_gt, mask, x):
    """
    BP-X: Percentage of pixels where disparity error is larger than X pixels.
    Args:
        disp_pred: predicted disparity (B, H, W)
        disp_gt: ground truth disparity (B, H, W)
        mask: valid mask (B, H, W)
        x: threshold in pixels
    Returns:
        Tensor of shape (B,) with percentage of bad pixels per image
    """
    E = torch.abs(disp_gt - disp_pred)
    err_mask = E > x
    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])
    bp_x_per_image = num_errors.float() / num_valid_pixels.float() * 100
    bp_x_per_image = torch.where(num_valid_pixels > 0, bp_x_per_image, torch.zeros_like(bp_x_per_image))
    return bp_x_per_image

def d1_metric(disp_pred, disp_gt, mask):
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)

    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    d1_per_image = num_errors.float() / num_valid_pixels.float() * 100
    d1_per_image = torch.where(num_valid_pixels > 0, d1_per_image, torch.zeros_like(d1_per_image))

    return d1_per_image


def threshold_metric(disp_pred, disp_gt, mask, threshold):
    E = torch.abs(disp_gt - disp_pred)
    err_mask = E > threshold

    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    bad_per_image = num_errors.float() / num_valid_pixels.float() * 100
    bad_per_image = torch.where(num_valid_pixels > 0, bad_per_image, torch.zeros_like(bad_per_image))

    return bad_per_image


def epe_metric(disp_pred, disp_gt, mask):
    E = torch.abs(disp_gt - disp_pred)
    E_masked = torch.where(mask, E, torch.zeros_like(E))

    E_sum = E_masked.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])
    epe_per_image = E_sum / num_valid_pixels
    epe_per_image = torch.where(num_valid_pixels > 0, epe_per_image, torch.zeros_like(epe_per_image))

    return epe_per_image
