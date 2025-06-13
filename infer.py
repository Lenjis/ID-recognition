import torch
from model import IDRecognizer
from utils import preprocess_image, transform
import os
import sys
from train import IDDataset, infer
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

INFER_MODE = "test" # train | test

TEST_PATH = f"data/{INFER_MODE}"  # 测试图像目录

if __name__ == "__main__":
    # infer(sys.argv[1])  # 命令行输入图像路径
    infer(TEST_PATH, INFER_MODE)
