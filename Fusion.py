import os
import glob
import math
import random

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# 融合模块
class AdditiveFusion(nn.Module):
    def __init__(self, F_v_channels, F_s_channels, out_channels):
        super(AdditiveFusion, self).__init__()
        # 确保通道数匹配
        self.conv_v = nn.Conv2d(F_v_channels, out_channels, 1)
        self.conv_s = nn.Conv2d(F_s_channels, out_channels, 1)
        self.activation = nn.ReLU(True)

    def forward(self, F_v, F_s):
        F_v = self.conv_v(F_v)
        F_s = self.conv_s(F_s)
        F_f = F_v + F_s
        F_f = self.activation(F_f)
        return F_f


class MultiplicativeFusion(nn.Module):
    def __init__(self, F_v_channels, F_s_channels, out_channels):
        super(MultiplicativeFusion, self).__init__()
        self.conv_v = nn.Conv2d(F_v_channels, out_channels, 1)
        self.conv_s = nn.Conv2d(F_s_channels, out_channels, 1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(True)

    def forward(self, F_v, F_s):
        F_v = self.conv_v(F_v)
        F_s = self.conv_s(F_s)
        F_f = F_v * F_s  # 元素级相乘
        F_f = self.gamma * F_f + self.beta
        F_f = self.activation(F_f)
        return F_f


class AttentionFusion(nn.Module):
    def __init__(self, F_v_channels, F_s_channels, out_channels, heads=4):
        super(AttentionFusion, self).__init__()
        self.conv_v = nn.Conv2d(F_v_channels, out_channels, 1)
        self.conv_s = nn.Conv2d(F_s_channels, out_channels, 1)
        self.attention = nn.MultiheadAttention(out_channels, heads)
        self.activation = nn.ReLU(True)

    def forward(self, F_v, F_s):
        F_v = self.conv_v(F_v)  # [B, C, H, W]
        F_s = self.conv_s(F_s)

        B, C, H, W = F_v.shape
        F_v = F_v.view(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]
        F_s = F_s.view(B, C, -1).permute(2, 0, 1)  # [H*W, B, C]

        att_output, _ = self.attention(F_v, F_s, F_s)  # [H*W, B, C]
        att_output = att_output.permute(1, 2, 0).view(B, C, H, W)  # [B, C, H, W]
        F_f = self.activation(att_output)
        return F_f