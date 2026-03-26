from pytorch_msssim import ssim
import torch

def ssim_loss(predicted, target, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    return 1 - ssim(predicted, target, win_size=win_size, win_sigma=win_sigma, K=K, data_range=1.0)

def l1_loss(predicted, target):
    return torch.mean(torch.abs(predicted - target))

def ssim_l1_loss(predicted, target, alpha=0.8, win_size=11, win_sigma=1.5, K=(0.01, 0.03)):
    ssim_val = 1 - ssim(predicted, target, win_size=win_size, win_sigma=win_sigma,
                        K=K, data_range=1.0)
    l1_val = torch.nn.functional.l1_loss(predicted, target)
    return alpha * ssim_val + (1 - alpha) * l1_val