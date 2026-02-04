from pytorch_msssim import ssim
import torch

def ssim_loss(predicted, target, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    return 1 - ssim(predicted, target, win_size=win_size, win_sigma=win_sigma, K=K, data_range=1.0)

def l1_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))