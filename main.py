import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import STL10
from torchvision.models.video import r3d_18, R3D_18_Weights
import matplotlib

matplotlib.use('Agg')

from tqdm import tqdm

# -----------------------------
# Part 1: Logging and Helper Functions
# -----------------------------
from Logger import Logger
from CheckFit import check_model_fit
from calculate import psnr_calculate, ssim_calculate, normalized_correlation

# -----------------------------
# Part 2: Dataset Definitions
# -----------------------------
from VideoClipDataset import VideoClipDataset

# -----------------------------
# Part 3: Model Definitions
# -----------------------------
# Generator
from Generator import Generator

class Discriminator3D(nn.Module):
    """
    改进的3D判别器 D，包括共享时空特征提取器 E_D 和隐写分析判别器 D_steg，
    其中 D_steg 使用预训练的 3D 卷积模型进行隐写分析，不使用注意力机制。
    """
    def __init__(self, video_in_channels=3, base_channels=32, temporal_depth=16):
        super(Discriminator3D, self).__init__()

        # 共享时空特征提取器 E_D
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(video_in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # [B, 3, T, 128, 128] -> [B, 32, T/2, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),  # -> [B, 64, T/4, 32, 32]
            nn.BatchNorm3d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # -> [B, 128, T/8, 16, 16]
            nn.BatchNorm3d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),  # -> [B, 256, T/16, 8, 8]
            nn.BatchNorm3d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 隐写分析判别器 D_steg，使用预训练的3D卷积模型
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
            validity_steg: [B], 隐写分析判别结果
        """
        # 提取共享特征
        F_D = self.feature_extractor(V)  # [B, 256, T/16, 8, 8]

        # 隐写分析
        # 由于预训练模型接受输入大小为 [B, 3, T, 112, 112]，需要调整输入
        # 这里假设原始视频的高度和宽度为128，将其调整为112
        # 使用三线性插值进行上采样
        V_resized = F.interpolate(V, size=(16, 112, 112), mode='trilinear', align_corners=False)  # [B, 3, 16, 112, 112]
        validity_steg = self.D_steg(V_resized).view(-1)  # [B]

        return validity_steg

# -----------------------------
# Part 4: 攻击函数定义
# -----------------------------
from Attack import apply_h264_compression, add_noise  # 导入攻击函数

# -----------------------------
# Part 5: 确定模型
# -----------------------------

def get_module(model):
    """
    Helper function to get the underlying module if wrapped with DataParallel
    """
    return model.module if isinstance(model, nn.DataParallel) else model

# -----------------------------
# Part 6: 训练函数定义（修改后的版本）
# -----------------------------
#====================================================
# (示例) 定义一个专门对 secret 图像做数据增强的函数
#====================================================
def augment_secret(secrets, config):
    """
    对 secret 图像进行数据增强。
    输入:
        secrets: [B, 3, H, W], 例如 [B, 3, 64, 64] 或 [B, 3, 96, 96]
        config:  配置字典，可定义各种数据增强超参数
    输出:
        secrets_augmented: [B, 3, H, W], 与输入形状相同
    """
    # 如果不需要数据增强，直接返回
    if not config.get('secret_augmentation', False):
        return secrets

    # 定义所需的图像增强操作 (示例：随机水平翻转 + 颜色扰动)
    augmentation_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=config.get('aug_brightness', 0.0),
            contrast=config.get('aug_contrast', 0.0),
            saturation=config.get('aug_saturation', 0.0),
            hue=config.get('aug_hue', 0.0)
        ),
    ])

    # 对每一张 secret 图像执行增强
    secrets_augmented = []
    for s in secrets:
        # s.shape = [3, H, W], 单张图像
        # 需把其转换成 [C, H, W] 的 Tensor 输入给 transforms
        # 由于 s 已经是 [3, H, W]，可直接使用。
        secrets_augmented.append(augmentation_transforms(s))

    # 堆叠回 batch
    secrets_augmented = torch.stack(secrets_augmented, dim=0)
    return secrets_augmented


def train(generator, discriminator, dataloader,
          optimizer_G, optimizer_D,
          criterion_BCE, criterion_MSE,
          device, logger, epoch, config):
    """
    视频隐写的训练过程。包括：
      1. 判别器 (D) 的对抗训练（区分正常视频 vs. 隐写视频）
      2. 生成器 (G) 的训练（在视频中嵌入并提取秘密信息，同时对抗判别器）
      3. 数据增强：对 secret 图像进行随机翻转、颜色扰动等，提升训练多样性。
    """

    generator.train()
    discriminator.train()

    total_D_loss = 0.0
    total_G_loss = 0.0
    total_L_steg = 0.0
    total_L_reg = 0.0
    steganalysis_correct = 0
    steganalysis_total = 0
    nc_before_list = []
    nc_after_list = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", unit="batch")
    for batch_idx, (frames, secret) in enumerate(pbar):
        # [B, T, 3, 128, 128], [B, 3, 64, 64] 等
        frames = frames.to(device)
        secret = secret.to(device)

        #====================
        # 对秘密图像进行数据增强
        #====================
        # 若配置中启用了 'secret_augmentation'，则对 secret 做随机翻转、颜色扰动等
        secret = augment_secret(secret, config).to(device)

        B, T, C, H, W = frames.shape

        # 调整帧维度以适应3D判别器 [B, 3, T, 128, 128]
        real_videos = frames.permute(0, 2, 1, 3, 4).contiguous()

        #====================================================
        # 1) 训练判别器 (Discriminator)
        #====================================================
        optimizer_D.zero_grad()

        # 判别器对正常视频的判断 (标签=0)
        stego_real = discriminator(real_videos)  # [B]

        # 生成隐写视频
        stego_frames, _ = generator(frames, secret)  # [B, T, 3, 128,128], [B, 3,64,64]
        fake_videos = stego_frames.permute(0, 2, 1, 3, 4).contiguous()
        stego_fake = discriminator(fake_videos.detach())  # [B]

        # 隐写分析损失 L_steg
        #   正常视频 → label=0
        #   隐写视频 → label=1
        labels_steg_real = torch.zeros_like(stego_real).to(device)
        labels_steg_fake = torch.ones_like(stego_fake).to(device)
        loss_steg_real = criterion_BCE(stego_real, labels_steg_real)
        loss_steg_fake = criterion_BCE(stego_fake, labels_steg_fake)
        L_steg = loss_steg_real + loss_steg_fake

        # 判别器正则化项 L_D_reg (示例使用 L2 范数)
        L_D_reg = sum(torch.norm(param, 2) for param in discriminator.parameters())

        # 判别器总损失
        Loss_D = config['lambda_steg'] * L_steg + config['lambda_D_reg'] * L_D_reg
        Loss_D.backward()
        optimizer_D.step()

        total_D_loss += Loss_D.item()
        total_L_steg += L_steg.item()
        total_L_reg += L_D_reg.item()

        #====================================================
        # 2) 训练生成器 (Generator)
        #====================================================
        optimizer_G.zero_grad()

        # 重新生成隐写视频
        stego_frames, secret_pred = generator(frames, secret)
        # 对隐写视频做模拟攻击 (H.264 编码压缩 + 噪声干扰)，提升鲁棒性
        stego_attacked = apply_h264_compression(stego_frames, device)
        stego_attacked = add_noise(stego_attacked, device, noise_std=config['noise_std'])

        # 从原始隐写视频和攻击后的视频中提取秘密信息
        _, secret_pred_attacked = generator(stego_attacked, None)

        # 计算秘密信息提取的损失 (在攻击前/后都要求接近原 secret)
        L_sec_before = criterion_MSE(secret_pred, secret)
        L_sec_after = criterion_MSE(secret_pred_attacked, secret)
        L_sec = L_sec_before + config['lambda_sec_after'] * L_sec_after

        # 生成器对抗损失，期望判别器将隐写视频判为“正常”
        fake_videos = stego_frames.permute(0, 2, 1, 3, 4).contiguous()
        stego_fake = discriminator(fake_videos)  # [B]
        labels_steg_real = torch.zeros_like(stego_fake).to(device)  # 目标标签设为0 (正常)
        G_adv_loss_steg = criterion_BCE(stego_fake, labels_steg_real)

        # 重建损失 L_recon：鼓励生成的视频接近原视频
        L_recon = F.l1_loss(stego_frames, frames)

        # 生成器总损失
        Loss_G = (
            config['lambda_steg_adv'] * G_adv_loss_steg
            + config['lambda_recon'] * L_recon
            + config['lambda_sec'] * L_sec
        )
        Loss_G.backward()
        optimizer_G.step()

        total_G_loss += Loss_G.item()

        #====================================================
        # 3) 隐写分析性能 (对抗准确率) 评估
        #====================================================
        # 正常视频=0, 隐写视频=1
        labels_steg = torch.cat([labels_steg_real, labels_steg_fake], dim=0)  # [2*B]
        outputs_steg = torch.cat([stego_real, stego_fake], dim=0)  # [2*B]
        predictions_steg = (outputs_steg > 0.5).float()
        steganalysis_correct += (predictions_steg == labels_steg).sum().item()
        steganalysis_total += labels_steg.size(0)

        #====================================================
        # 4) 计算 NC 指标 (Normalized Correlation)
        #====================================================
        # (a) 在无攻击场景下的提取
        nc_before = normalized_correlation(secret_pred, secret)  # [B]
        nc_before_avg = nc_before.mean().item() if isinstance(nc_before, torch.Tensor) else float(nc_before)
        nc_before_list.append(nc_before_avg)

        # (b) 在攻击场景下的提取
        nc_after = normalized_correlation(secret_pred_attacked, secret)  # [B]
        nc_after_avg = nc_after.mean().item() if isinstance(nc_after, torch.Tensor) else float(nc_after)
        nc_after_list.append(nc_after_avg)

        # 更新进度条显示
        pbar.set_postfix({
            'Loss_D': f"{Loss_D.item():.3f}",
            'Loss_G': f"{Loss_G.item():.3f}",
            'L_steg': f"{L_steg.item():.3f}",
            'L_reg': f"{L_D_reg.item():.3f}",
            'G_adv_loss_steg': f"{G_adv_loss_steg.item():.3f}",
            'L_recon': f"{L_recon.item():.3f}",
            'L_sec': f"{L_sec.item():.3f}",
            'StegAcc': f"{steganalysis_correct / steganalysis_total:.3f}" if steganalysis_total>0 else 0,
            'NC_before': f"{nc_before_avg:.3f}",
            'NC_after': f"{nc_after_avg:.3f}",
        })

    #====================================================
    # 5) 统计本轮 (epoch) 的平均指标并输出日志
    #====================================================
    avg_D_loss = total_D_loss / len(dataloader)
    avg_G_loss = total_G_loss / len(dataloader)
    avg_L_steg = total_L_steg / len(dataloader)
    avg_L_reg = total_L_reg / len(dataloader)
    steganalysis_accuracy = steganalysis_correct / steganalysis_total if steganalysis_total > 0 else 0
    avg_nc_before = np.mean(nc_before_list) if nc_before_list else 0
    avg_nc_after = np.mean(nc_after_list) if nc_after_list else 0

    logger.info(
        f"Epoch [{epoch+1}/{config['num_epochs']}], "  
        f"Loss_D: {avg_D_loss:.4f}, Loss_G: {avg_G_loss:.4f}, "  
        f"L_steg: {avg_L_steg:.4f}, L_reg: {avg_L_reg:.4f}, "  
        f"StegAcc: {steganalysis_accuracy:.4f}, "  
        f"NC_before: {avg_nc_before:.4f}, NC_after: {avg_nc_after:.4f}"
    )

    return avg_D_loss, avg_G_loss, steganalysis_accuracy, avg_nc_before, avg_nc_after

# -----------------------------
# Part 7: 验证函数定义（修改后的版本）
# -----------------------------
def validate(generator, discriminator, dataloader, device, logger, epoch, config):
    import torchvision.utils as vutils
    generator.eval()
    discriminator.eval()

    psnr_list = []
    ssim_list = []
    nc_before_list = []
    nc_after_list = []
    steganalysis_correct = 0
    steganalysis_total = 0

    # 创建保存可视化结果的目录
    save_results_dir = os.path.join(config['save_dir'], f'epoch_{epoch+1}')
    os.makedirs(save_results_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (frames, secret) in enumerate(dataloader):
            frames = frames.to(device)  # [B, T, 3, 128, 128]
            secret = secret.to(device)  # [B, 3, 64, 64]
            B, T, C, H, W = frames.shape

            # 生成隐写视频和提取的秘密信息
            stego_frames, secret_pred = generator(frames, secret)  # [B, T, 3, 128,128], [B, 3,64,64]

            # 对隐写视频应用攻击
            stego_attacked = apply_h264_compression(stego_frames, device)  # [B, T, 3, H, W]
            stego_attacked = add_noise(stego_attacked, device, noise_std=config['noise_std'])  # 添加噪声

            # 从攻击后的隐写视频中提取秘密信息
            _, secret_pred_attacked = generator(stego_attacked, None)  # [B, 3,64,64]

            # 多样性评价：NC值计算
            # 提取秘密信息前
            nc_before = normalized_correlation(secret_pred, secret)  # [B]
            if isinstance(nc_before, torch.Tensor):
                nc_before_avg = nc_before.mean().item()
            else:
                nc_before_avg = float(nc_before)
            nc_before_list.append(nc_before_avg)

            # 提取秘密信息后
            nc_after = normalized_correlation(secret_pred_attacked, secret)  # [B]
            if isinstance(nc_after, torch.Tensor):
                nc_after_avg = nc_after.mean().item()
            else:
                nc_after_avg = float(nc_after)
            nc_after_list.append(nc_after_avg)

            # 计算PSNR和SSIM
            stego_videos = stego_frames.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, 128, 128]
            real_videos = frames.permute(0, 2, 1, 3, 4).contiguous()        # [B, 3, T, 128, 128]
            for b in range(B):
                for t in range(T):
                    psnr_val = psnr_calculate(stego_videos[b, :, t], real_videos[b, :, t])
                    ssim_val = ssim_calculate(stego_videos[b, :, t], real_videos[b, :, t])
                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)

            # 隐写分析性能评估
            # 标签：正常视频为0，隐写视频为1
            real_videos_flat = frames.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, 128, 128]
            stego_videos_attacked = stego_attacked.permute(0, 2, 1, 3, 4).contiguous()  # [B, 3, T, 128, 128]
            stego_real = discriminator(real_videos_flat)  # [B]
            stego_fake = discriminator(stego_videos_attacked)  # [B]

            labels_steg_real = torch.zeros_like(stego_real).to(device)  # 正常视频标签为0
            labels_steg_fake = torch.ones_like(stego_fake).to(device)   # 隐写视频标签为1
            labels_steg = torch.cat([labels_steg_real, labels_steg_fake], dim=0)  # [2*B]

            outputs_steg = torch.cat([stego_real, stego_fake], dim=0)  # [2*B]
            predictions_steg = (outputs_steg > 0.5).float()
            steganalysis_correct += (predictions_steg == labels_steg).sum().item()
            steganalysis_total += labels_steg.size(0)

            # ---------------------
            # 保存可视化结果
            # ---------------------
            # 仅保存第一个batch，或者根据需要修改
            if batch_idx == 0:
                num_samples = min(B, config.get('num_visualization_samples', 4))
                for i in range(num_samples):
                    # 保存原始视频帧
                    original_frames = frames[i]  # [T, 3, H, W]
                    os.makedirs(os.path.join(save_results_dir, f'sample_{i}', 'original_frames'), exist_ok=True)
                    for t in range(T):
                        frame = original_frames[t]
                        vutils.save_image(frame, os.path.join(save_results_dir, f'sample_{i}', 'original_frames', f'frame_{t}.png'), normalize=True)

                    # 保存隐写视频帧
                    stego_frames_i = stego_frames[i]  # [T, 3, H, W]
                    os.makedirs(os.path.join(save_results_dir, f'sample_{i}', 'stego_frames'), exist_ok=True)
                    for t in range(T):
                        frame = stego_frames_i[t]
                        vutils.save_image(frame, os.path.join(save_results_dir, f'sample_{i}', 'stego_frames', f'frame_{t}.png'), normalize=True)

                    # 保存攻击后的隐写视频帧
                    stego_attacked_i = stego_attacked[i]  # [T, 3, H, W]
                    os.makedirs(os.path.join(save_results_dir, f'sample_{i}', 'stego_attacked_frames'), exist_ok=True)
                    for t in range(T):
                        frame = stego_attacked_i[t]
                        vutils.save_image(frame, os.path.join(save_results_dir, f'sample_{i}', 'stego_attacked_frames', f'frame_{t}.png'), normalize=True)

                    # 保存原始秘密图像
                    secret_i = secret[i]  # [3, 64, 64]
                    os.makedirs(os.path.join(save_results_dir, f'sample_{i}'), exist_ok=True)
                    vutils.save_image(secret_i, os.path.join(save_results_dir, f'sample_{i}', 'secret.png'), normalize=True)

                    # 保存未攻击时提取的秘密图像
                    secret_pred_i = secret_pred[i]  # [3, 64, 64]
                    vutils.save_image(secret_pred_i, os.path.join(save_results_dir, f'sample_{i}', 'secret_pred.png'), normalize=True)

                    # 保存攻击后提取的秘密图像
                    secret_pred_attacked_i = secret_pred_attacked[i]  # [3, 64, 64]
                    vutils.save_image(secret_pred_attacked_i, os.path.join(save_results_dir, f'sample_{i}', 'secret_pred_attacked.png'), normalize=True)

    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    avg_nc_before = np.mean(nc_before_list) if nc_before_list else 0
    avg_nc_after = np.mean(nc_after_list) if nc_after_list else 0
    steganalysis_accuracy = steganalysis_correct / steganalysis_total if steganalysis_total > 0 else 0

    logger.info(
        f"Validation Epoch [{epoch+1}/{config['num_epochs']}], "
        f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, "
        f"NC_before Attack: {avg_nc_before:.4f}, NC_after Attack: {avg_nc_after:.4f}, "
        f"Steganalysis Accuracy: {steganalysis_accuracy:.4f}"
    )

    return avg_psnr, avg_ssim, avg_nc_before, avg_nc_after, steganalysis_accuracy

def initialize(config): # 初始化函数
    # 确保目录存在
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['metrics_dir'], exist_ok=True)

    # 初始化日志
    logger = Logger(log_dir=config['log_dir'])
    logger.info("开始训练过程")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'使用设备: {device}')

    # 设置随机种子（可选）
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    # 数据增强（如果需要可以添加）
    transform_secret = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        # 秘密图像保持在 [0,1] 范围，无需归一化
    ])

    # 加载秘密信息数据集（STL-10）
    secret_dataset = STL10(root='./data', split='train', download=True, transform=transform_secret)

    # 创建视频隐写数据集
    dataset = VideoClipDataset(
        video_dir=config['video_dir'],
        secret_dataset=secret_dataset,
        target_fps=30,  # 目标FPS
        duration=1.0,  # 隐写的持续时间（前3秒）
        transform=None,  # 如有需要，可添加视频帧的transform
        secret_transform=None  # 已在Dataset中处理
    )

    # 划分训练集和验证集
    total_size = len(dataset)
    val_size = config['validation_size']
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    return logger, device, train_loader, val_loader
# -----------------------------
# Part 8: 主程序
# -----------------------------
def main():
    # 配置参数
    from Config import config  # 参数设置及设备日志初始化

    # 初始化
    logger, device, train_loader, val_loader = initialize(config)

    # 创建生成器和判别器
    generator = Generator(
        video_in_channels=3,
        secret_in_channels=3,
        base_channels=64
    ).to(device)

    discriminator = Discriminator3D(
        video_in_channels=3,
        base_channels=32,
        temporal_depth=config.get('temporal_depth', 16)  # 从配置中获取时序深度
    ).to(device)

    # 初始化权重
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    generator.apply(weights_init_normal)
    # 对预训练的判别器，不要重新初始化其权重

    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

    # 学习率调度器
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

    # 损失函数
    criterion_BCE = nn.BCELoss().to(device)
    criterion_MSE = nn.MSELoss().to(device)

    # 训练历史
    metrics = {
        'loss_D': [],
        'loss_G': [],
        'steganalysis_acc': [],
        'nc_before': [],
        'nc_after': [],
        'psnr': [],
        'ssim': []
    }

    # 训练循环
    for epoch in range(config['num_epochs']):
        # 训练
        avg_D_loss, avg_G_loss, steganalysis_acc, avg_nc_before, avg_nc_after = train(
            generator, discriminator, train_loader, optimizer_G, optimizer_D,
            criterion_BCE, criterion_MSE, device, logger, epoch, config
        )
        metrics['loss_D'].append(avg_D_loss)
        metrics['loss_G'].append(avg_G_loss)
        metrics['steganalysis_acc'].append(steganalysis_acc)
        metrics['nc_before'].append(avg_nc_before)
        metrics['nc_after'].append(avg_nc_after)

        # 验证
        avg_psnr, avg_ssim, avg_nc_before_val, avg_nc_after_val, steganalysis_acc_val = validate(
            generator, discriminator, val_loader, device, logger, epoch, config
        )
        metrics['psnr'].append(avg_psnr)
        metrics['ssim'].append(avg_ssim)
        metrics['nc_before'].append(avg_nc_before_val)
        metrics['nc_after'].append(avg_nc_after_val)
        metrics['steganalysis_acc'].append(steganalysis_acc_val)

        # 保存指标到文件
        try:
            with open(os.path.join(config['metrics_dir'], 'loss_D.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_D_loss}\n")
            with open(os.path.join(config['metrics_dir'], 'loss_G.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_G_loss}\n")
            with open(os.path.join(config['metrics_dir'], 'steganalysis_acc.txt'), 'a') as f:
                f.write(f"{epoch+1},{steganalysis_acc_val}\n")
            with open(os.path.join(config['metrics_dir'], 'nc_before.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_nc_before_val}\n")
            with open(os.path.join(config['metrics_dir'], 'nc_after.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_nc_after_val}\n")
            with open(os.path.join(config['metrics_dir'], 'psnr.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_psnr}\n")
            with open(os.path.join(config['metrics_dir'], 'ssim.txt'), 'a') as f:
                f.write(f"{epoch+1},{avg_ssim}\n")
        except Exception as e:
            logger.error(f"Error writing metrics to file: {e}")

        # 更新训练历史中的部分指标（用于检查模型拟合）
        # 注意：这里的 'nc_before' 和 'nc_after' 同时记录了训练和验证的值，
        # 如果需要区分，可以调整 metrics 字典结构。

        # 每隔一定周期检查模型拟合情况
        if (epoch + 1) % config['check_fit_interval'] == 0:
            check_model_fit(
                metrics['loss_D'], metrics['loss_G'],
                metrics['steganalysis_acc'],
                metrics['nc_before'], metrics['nc_after'],
                epoch, logger
            )

        # 调整学习率
        scheduler_G.step()
        scheduler_D.step()

        # 每隔一定周期保存模型
        if (epoch + 1) % config['save_interval'] == 0:
            try:
                torch.save(get_module(generator).state_dict(), os.path.join(config['save_dir'], f'generator_epoch_{epoch+1}.pth'))
                torch.save(get_module(discriminator).state_dict(), os.path.join(config['save_dir'], f'discriminator_epoch_{epoch+1}.pth'))
            except Exception as e:
                logger.error(f"Error saving model at epoch {epoch+1}: {e}")

    # 保存最终模型
    try:
        torch.save(get_module(generator).state_dict(), os.path.join(config['save_dir'], 'generator_final.pth'))
        torch.save(get_module(discriminator).state_dict(), os.path.join(config['save_dir'], 'discriminator_final.pth'))
    except Exception as e:
        logger.error(f"Error saving final models: {e}")

    logger.info("训练完成")

if __name__ == '__main__':
    main()