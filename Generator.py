import torch
import torch.nn as nn
import torch.nn.functional as F


########################################################
# 新增 1: 定义Squeeze-and-Excitation (SE) 3D注意力模块
########################################################
class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation for 3D特征图：
    - 自适应地为每个通道分配权重，从而让网络更关注关键信息。
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock3D, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        # 通道维度全局平均池化
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


########################################################
# 1. 3D加法融合模块（原样保持）
########################################################
class AdditiveFusion3D(nn.Module):
    def __init__(self, in_channels_v, in_channels_s, out_channels):
        super(AdditiveFusion3D, self).__init__()
        self.conv_v = nn.Conv3d(in_channels_v, out_channels, kernel_size=1)
        self.conv_s = nn.Conv3d(in_channels_s, out_channels, kernel_size=1)

    def forward(self, feat_v, feat_s):
        return self.conv_v(feat_v) + self.conv_s(feat_s)


########################################################
# 2. 改进版 3D U-Net 用于秘密信息提取
#    - 引入 SEBlock3D，提高对关键信息的关注
########################################################
class SecretExtractor3D(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, out_channels=3):
        super(SecretExtractor3D, self).__init__()

        # 编码器部分
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=(3, 4, 4),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels)  # 在每层Enc后加入SE
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 4, 4),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels * 2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 4, 4),
                      stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels * 4)
        )

        # 解码器部分
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                               kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels * 2)
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels,
                               kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 2, out_channels,
                               kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, 3, T, 128,128] 含隐写信息的视频序列
        返回:
          [B, out_channels, T, 128,128]
        """
        e1 = self.enc1(x)   # => [B, 64,  T, 64,64]
        e2 = self.enc2(e1)  # => [B,128, T, 32,32]
        e3 = self.enc3(e2)  # => [B,256, T,16,16]

        d3 = self.dec3(e3)  # => [B,128, T,32,32]
        d3_cat = torch.cat([d3, e2], dim=1)  # => [B,256, T,32,32]
        d2 = self.dec2(d3_cat)               # => [B,64,  T,64,64]
        d2_cat = torch.cat([d2, e1], dim=1)  # => [B,128, T,64,64]
        d1 = self.dec1(d2_cat)               # => [B,3,   T,128,128]
        return d1


########################################################
# 3. 改进后的生成器模型
#    - 在视频编码器增加通道注意力(SEBlock3D)
#    - 其他逻辑与原代码基本一致
########################################################
class Generator(nn.Module):
    """
    在保持不可见性的基础上，尽可能提升秘密信息的提取性能与嵌入容量:
      1) 视频编码器加入 SEBlock3D 增强通道注意力
      2) 保持大体结构不变，去除 Tanh 直接输出扰动
      3) 可学习 alpha 控制嵌入强度
      4) 三层 3D U-Net 提取秘密信息
    """

    def __init__(self,
                 video_in_channels=3,
                 secret_in_channels=3,
                 base_channels=64,
                 embed_frame_count=30):
        super(Generator, self).__init__()
        self.embed_frame_count = embed_frame_count

        ##############################################
        # (A) 视频编码器（3D卷积 + SEBlock3D）
        ##############################################
        self.E_v3d = nn.Sequential(
            nn.Conv3d(video_in_channels, base_channels, kernel_size=(3, 4, 4),
                      stride=(1, 2, 2), padding=(1, 1, 1)),  # 128->64
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels),

            nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 4, 4),
                      stride=(1, 2, 2), padding=(1, 1, 1)),  # 64->32
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels * 2),

            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 4, 4),
                      stride=(1, 2, 2), padding=(1, 1, 1)),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels * 4),
        )
        # => [B, base_channels*4, T, 16,16]

        ##############################################
        # (B) 秘密信息编码器（2D）：只做2次下采样
        ##############################################
        self.E_s = nn.Sequential(
            nn.Conv2d(secret_in_channels, base_channels, kernel_size=4,
                      stride=2, padding=1),  # 96->48
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4,
                      stride=2, padding=1),  # 48->24
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        # => [B, base_channels*2, 24,24]

        ##############################################
        # (C) 3D融合模块 (加法形式)
        ##############################################
        self.fusion3d = AdditiveFusion3D(
            in_channels_v=base_channels * 4,
            in_channels_s=base_channels * 2,
            out_channels=base_channels * 4
        )

        ##############################################
        # (D) 3D解码器：生成秘密嵌入扰动 (去掉 Tanh)
        ##############################################
        self.D3d = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                               kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels * 2),

            nn.ConvTranspose3d(base_channels * 2, base_channels,
                               kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(inplace=True),
            SEBlock3D(base_channels),

            nn.ConvTranspose3d(base_channels, video_in_channels,
                               kernel_size=(3, 4, 4),
                               stride=(1, 2, 2), padding=(1, 1, 1))
            # 不做Tanh，直接输出 (可后续 clamp)
        )

        ##############################################
        # (E) 3D秘密信息提取模块 (含注意力的三层U-Net)
        ##############################################
        self.R3d = SecretExtractor3D(
            in_channels=video_in_channels,
            base_channels=base_channels,
            out_channels=secret_in_channels
        )

        ##############################################
        # (F) 可学习的残差权重 alpha
        ##############################################
        self.alpha = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, frames, secret=None):
        """
        参数:
            frames: [B, T, 3, 128,128] - 输入视频帧
            secret: [B, 3, 96,96]     - 输入秘密图像
        返回:
            stego_frames: [B, T, 3, 128,128] - 融合秘密后的视频
            secret_pred:  [B, 3, 96,96]     - 提取出的秘密图像
        """
        B, T, C, H, W = frames.shape
        frames_3d = frames.transpose(1, 2).contiguous()  # => [B, 3, T, 128,128]

        if secret is not None:
            # ========== 1) 对可嵌入帧进行处理 ==========
            T_embed = min(T, self.embed_frame_count)
            frames_embed = frames_3d[:, :, :T_embed, :, :]  # => [B, 3, T_embed, 128,128]

            # ========== 2) 视频特征提取 ==========
            F_v = self.E_v3d(frames_embed)  # => [B, 4*base_channels, T_embed, 16,16]

            # ========== 3) 秘密信息编码(2D) ==========
            F_s_2d = self.E_s(secret)       # => [B, 2*base_channels, 24,24]
            # 插值到 [16,16]
            F_s_2d = F.interpolate(F_s_2d, size=(16, 16),
                                   mode='bilinear', align_corners=False)
            # 扩展到3D: [B, 2*base_channels, T_embed, 16,16]
            F_s_3d = F_s_2d.unsqueeze(2).repeat(1, 1, T_embed, 1, 1)

            # ========== 4) 融合特征(3D加法) ==========
            F_f = self.fusion3d(F_v, F_s_3d)  # => [B, 4*base_channels, T_embed, 16,16]

            # ========== 5) 生成扰动 + 残差叠加 ==========
            perturbation = self.D3d(F_f)  # => [B, 3, T_embed, 128,128]
            stego_embed = frames_embed + self.alpha * perturbation
            stego_embed = torch.clamp(stego_embed, -1, 1)

            # ========== 6) 提取秘密信息并插值回原分辨率(96×96) ==========
            secret_preds_3d = self.R3d(stego_embed)  # => [B, 3, T_embed, 128,128]
            # 简单地对时间维度做平均，可根据需要改为取第0帧或其他策略
            secret_pred_2d = secret_preds_3d.mean(dim=2)  # => [B, 3, 128,128]
            secret_pred_2d = F.interpolate(secret_pred_2d,
                                           size=(96, 96),
                                           mode='bilinear',
                                           align_corners=False)

            # ========== 7) 拼接回完整视频 ==========
            stego_embed_frames = stego_embed.transpose(1, 2).contiguous()  # => [B, T_embed, 3, 128,128]
            if T_embed < T:
                stego_tail = frames[:, T_embed:, :, :, :]
                stego_frames = torch.cat([stego_embed_frames, stego_tail], dim=1)
            else:
                stego_frames = stego_embed_frames

            return stego_frames, secret_pred_2d
        else:
            # ========== 仅提取秘密信息 (无secret输入) ==========
            T_extract = min(T, self.embed_frame_count)
            frames_extract = frames_3d[:, :, :T_extract, :, :]

            secret_preds_3d = self.R3d(frames_extract)  # => [B, 3, T_extract, 128,128]
            secret_pred_2d = secret_preds_3d.mean(dim=2)  # => [B, 3, 128,128]
            secret_pred_2d = F.interpolate(secret_pred_2d,
                                           size=(96, 96),
                                           mode='bilinear',
                                           align_corners=False)

            return frames, secret_pred_2d