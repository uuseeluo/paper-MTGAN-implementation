# calculate.py

import math
import torch
import torch.nn.functional as F

def psnr_calculate(img1, img2):
    """
    计算 PSNR 值, img1, img2: [C,H,W] 或 [B,C,H,W], 取值范围 [0,1].
    """
    if img1.dim() == 4:
        mse = F.mse_loss(img1, img2, reduction='mean').item()
    else:
        mse = F.mse_loss(img1, img2, reduction='mean').item()

    if mse == 0:
        return 100.0
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def ssim_calculate(img1, img2):
    """
    计算 SSIM 值的简化版本, img1, img2: [C,H,W], 取值范围 [0,1].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    if img1.dim() == 4:
        img1 = img1[0]
        img2 = img2[0]

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denom = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)
    ssim /= (denom + 1e-8)
    return ssim.item()


def normalized_correlation(secret_pred, secret_gt):
    """
    计算 NC（Normalized Correlation）
    secret_pred, secret_gt: [C,H,W] 或 [B,C,H,W]
    """
    if secret_pred.dim() == 4:
        secret_pred = secret_pred[0]
        secret_gt = secret_gt[0]
    secret_pred = secret_pred.view(-1).float()
    secret_gt = secret_gt.view(-1).float()

    numerator = torch.sum(secret_pred * secret_gt)
    denominator = torch.sqrt(torch.sum(secret_pred ** 2)) * torch.sqrt(torch.sum(secret_gt ** 2))
    nc_value = numerator / (denominator + 1e-8)
    return nc_value.item()