import os
import glob
import random

import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


class VideoClipDataset(Dataset):
    """
    Extracts the first 0.5 seconds of frames from videos to serve as cover video frames and embeds secret information uniformly into these frames.
    """
    def __init__(self, video_dir, secret_dataset, target_fps=30, duration=1.0, transform=None, secret_transform=None):
        """
        参数:
            video_dir: 视频文件夹路径，支持通配符（glob）加载
            secret_dataset: 随机选择的秘密图像数据集
            target_fps: 目标帧率，默认30 FPS
            duration: 视频片段时长，单位秒（此处设为0.5秒）
            transform: 针对视频帧序列的预处理变换
            secret_transform: 针对秘密图像的预处理变换
        """
        super(VideoClipDataset, self).__init__()
        self.video_paths = glob.glob(video_dir, recursive=True)
        if len(self.video_paths) == 0:
            raise ValueError("No video files found in the specified directory pattern.")

        self.target_fps = target_fps
        self.duration = duration
        self.fixed_length = int(target_fps * duration)  # Number of frames in the specified duration (0.5秒)
        self.transform = transform
        self.secret_dataset = secret_dataset
        self.secret_transform = secret_transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        secret_img, _ = random.choice(self.secret_dataset)  # Randomly select a secret image

        # Extract the first 0.5 seconds of frames
        frames = self._extract_fixed_length_frames(video_path)  # [fixed_length, 3, 128, 128]

        if self.transform is not None:
            frames = self.transform(frames)

        if self.secret_transform is not None:
            secret_img = self.secret_transform(secret_img)

        return frames, secret_img

    def _extract_fixed_length_frames(self, video_path):
        """
        Extracts a fixed number of frames from the first 0.5 seconds of the video, padding with zero frames if necessary.
        """
        cap = cv2.VideoCapture(video_path)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps == 0:
            actual_fps = self.target_fps  # Avoid division by zero

        # Calculate the total number of frames to read (e.g. 30 FPS * 0.5秒 = 15帧)
        frames_to_read = self.fixed_length

        frames_list = []
        for i in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                # Pad with zero frames if insufficient frames
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_LINEAR)
            frames_list.append(frame)

        cap.release()

        frames_array = np.array(frames_list)  # [fixed_length, 128, 128, 3]
        frames_tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float() / 255.0  # [fixed_length, 3, 128, 128]

        return frames_tensor