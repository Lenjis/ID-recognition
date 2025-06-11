import cv2
import numpy as np
from torchvision import transforms


# 图像处理流程：灰度化 + 二值化 + resize
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.resize(binary, (200, 50))  # 例：10位学号，每位宽20，高50
    return binary.astype(np.float32) / 255.0


# PyTorch transform
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # [H, W] → [1, H, W]
    ]
)
