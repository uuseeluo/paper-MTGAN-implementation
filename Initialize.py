import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import STL10
from Logger import Logger  # 假设您有一个自定义的Logger类


