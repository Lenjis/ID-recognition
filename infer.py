import torch
from model import IDRecognizer
from utils import preprocess_image, transform
import os
import sys

TEST_PATH = "data/test"  # 测试图像目录

def infer(img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDRecognizer().to(device)
    model.load_state_dict(torch.load("id_model.pth", map_location=device))
    model.eval()

    img = preprocess_image(img_path)
    img = transform(img).unsqueeze(0).to(device)  # [1, 1, H, W]

    with torch.no_grad():
        outputs = model(img)
        preds = [str(torch.argmax(logit, dim=1).item()) for logit in outputs]

    print("Predicted ID:", "".join(preds))


if __name__ == "__main__":
    # infer(sys.argv[1])  # 命令行输入图像路径
    for img in os.listdir(TEST_PATH):
        img_path = os.path.join(TEST_PATH, img)
        infer(img_path)
