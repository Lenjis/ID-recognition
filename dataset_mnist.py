import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

# 参数配置
NUM_SAMPLES = 100  # 生成样本数量
DIGITS_PER_ID = 14  # 学号位数
# OUTPUT_DIR = 'data/images'
OUTPUT_DIR = "data/test"
# LABEL_FILE = 'data/labels.txt'
LABEL_FILE = "data/test_labels.txt"
IMG_HEIGHT = 28
IMG_WIDTH = 28 * DIGITS_PER_ID

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 MNIST
mnist = torchvision.datasets.MNIST(root="data/mnist/", train=True, download=True)

# 构建数字到图像的映射（0~9）
digit_map = {i: [] for i in range(10)}
for img, label in mnist:
    digit_map[label].append(img)

# 生成数据集
with open(LABEL_FILE, "w") as f:
    for idx in tqdm(range(NUM_SAMPLES)):
        digits = np.random.randint(0, 10, size=DIGITS_PER_ID)
        img_pieces = [
            digit_map[int(d)][np.random.randint(len(digit_map[int(d)]))] for d in digits
        ]
        id_img = Image.new("L", (IMG_WIDTH, IMG_HEIGHT))  # 拼接后的图像
        for i, digit_img in enumerate(img_pieces):
            id_img.paste(digit_img, (i * 28, 0))
        filename = f"img_{idx:04d}.jpg"
        id_img.save(os.path.join(OUTPUT_DIR, filename))
        f.write(f"{filename} {''.join(map(str, digits))}\n")
