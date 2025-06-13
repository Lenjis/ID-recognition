import cv2
import numpy as np
from PIL import Image
from torchvision import transforms as T


# 图像处理流程：灰度化 + 二值化 + resize
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # binary = cv2.resize(binary, (200, 50))  # 例：10位学号，每位宽20，高50
    # return binary.astype(np.float32) / 255.0
    pil_img = Image.fromarray(img)   # 转为PIL.Image
    return pil_img


# 图像变换
transform = T.Compose(
    [
        T.RandomRotation(2), # 随机旋转
        T.RandomAffine(0, translate=(0.05, 0.05)),  # 随机平移
        T.ToTensor(),
    ]
)  # [H, W] → [1, H, W]
