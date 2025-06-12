import os

import numpy as np
import torch

def apply_h264_compression(video_frames, device):
    """
    对视频帧应用 H.264 压缩和解压缩攻击，使用 PyAV（基于 FFmpeg 的 Python 库）进行内存中的操作。
    video_frames: [B, T, 3, H, W] tensor
    Returns:
        attacked_frames: [B, T, 3, H, W] tensor
    """
    import av
    B, T, C, H, W = video_frames.size()
    attacked_frames = []

    for b in range(B):
        # 创建一个内存中的字节缓冲区
        import io
        buffer = io.BytesIO()

        # 创建输出容器，使用 PyAV
        output_container = av.open(buffer, mode='w', format='mp4')

        # 添加一个视频流，指定编码器为 H.264
        stream = output_container.add_stream('h264', rate=30)  # 假设帧率为30
        stream.width = W
        stream.height = H
        stream.pix_fmt = 'yuv420p'

        # 编码帧
        for t in range(T):
            frame = video_frames[b, t].permute(1, 2, 0).cpu().detach().numpy()  # [H, W, 3]
            frame = (frame * 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                output_container.mux(packet)

        # 刷新编码器缓冲区
        for packet in stream.encode():
            output_container.mux(packet)

        output_container.close()

        buffer.seek(0)

        # 使用 PyAV 从压缩的视频中解码帧
        input_container = av.open(buffer, mode='r', format='mp4')

        decoded_frames = []
        for frame in input_container.decode(video=0):
            img = frame.to_rgb().to_ndarray()  # [H, W, 3]
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]
            decoded_frames.append(img)

        input_container.close()

        # 如果解码的帧数少于原始帧数，重复最后一帧
        if len(decoded_frames) < T:
            last_frame = decoded_frames[-1]
            for _ in range(T - len(decoded_frames)):
                decoded_frames.append(last_frame)

        # 如果解码的帧数多于原始帧数，截断多余的帧
        decoded_frames = decoded_frames[:T]

        decoded_frames = torch.stack(decoded_frames)  # [T, 3, H, W]
        attacked_frames.append(decoded_frames)

    attacked_frames = torch.stack(attacked_frames).to(device)  # [B, T, 3, H, W]

    # 转换形状以匹配输入
    return attacked_frames

def add_noise(video_frames, device, noise_std=0.02):
    """
    对视频帧添加高斯噪声攻击。
    video_frames: [B, T, 3, H, W] tensor
    Returns:
        noised_frames: [B, T, 3, H, W] tensor
    """
    noise = torch.randn_like(video_frames) * noise_std
    noised_frames = video_frames + noise
    noised_frames = torch.clamp(noised_frames, 0.0, 1.0)
    return noised_frames