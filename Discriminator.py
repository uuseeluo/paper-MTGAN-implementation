import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

# -----------------------------
# Part 3.1: 修改后的判别器定义，使用预训练的3D卷积模型进行隐写分析
# -----------------------------
class Discriminator3D(nn.Module):
    """
    改进的3D判别器 D，包括共享时空特征提取器 E_D、视频真伪判别器 D_real 和隐写分析判别器 D_steg，
    其中 D_steg 使用预训练的 3D 卷积模型进行隐写分析，不使用注意力机制。
    """
    def __init__(self, video_in_channels=3, base_channels=32, temporal_depth=16):
        super(Discriminator3D, self).__init__()

        # 共享时空特征提取器 E_D
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(video_in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # [B, 3, T, 128, 128] -> [B, 64, T/2, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # -> [B, 128, T/4, 32, 32]
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # -> [B, 256, T/8, 16, 16]
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),  # -> [B, 512, T/16, 8, 8]
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 视频真伪判别器 D_real
        self.D_real = nn.Sequential(
            nn.Conv3d(base_channels * 8, 1, kernel_size=(temporal_depth // 16, 8, 8), stride=1),  # 输出单个值
            nn.Sigmoid()
        )

        # 隐写分析判别器 D_steg，使用预训练的3D卷积模型
        # 这里使用 ResNet3D-18 预训练模型
        # 使用KINETICS400_V1预训练权重
        self.D_steg = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        # 修改全连接层
        self.D_steg.fc = nn.Sequential(
            nn.Linear(self.D_steg.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, V):
        """
        V: [B, 3, T, 128, 128]
        Returns:
            validity_real: [B], 视频真伪判别结果
            validity_steg: [B], 隐写分析判别结果
        """
        # 提取共享特征
        F_D = self.feature_extractor(V)  # [B, 512, T/16, 8, 8]

        # 视频真伪判别
        validity_real = self.D_real(F_D).view(-1)  # [B]

        # 隐写分析
        # 由于预训练模型接受输入大小为 [B, 3, T, 112, 112]，需要调整输入
        # 这里假设原始视频的高度和宽度为128，将其调整为112
        # 使用三线性插值进行上采样
        V_resized = F.interpolate(V, size=(16, 112, 112), mode='trilinear', align_corners=False)  # [B, 3, 16, 112, 112]
        validity_steg = self.D_steg(V_resized).view(-1)  # [B]

        return validity_real, validity_steg