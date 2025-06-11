import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import IDRecognizer
from utils import preprocess_image, transform
import os


class IDDataset(Dataset):
    def __init__(self, img_dir, label_file):
        self.img_dir = img_dir
        self.samples = []
        with open(label_file) as f:
            for line in f:
                img_name, label = line.strip().split()
                self.samples.append((img_name, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img = preprocess_image(os.path.join(self.img_dir, img_name))
        img = transform(img)  # [1, H, W]
        label = torch.tensor([int(c) for c in label], dtype=torch.long)
        return img, label
    

# 训练主函数
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDRecognizer().to(device)
    dataset = IDDataset("data/images", "data/labels.txt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)  # list of logits

            loss = sum(
                [criterion(logits, labels[:, i]) for i, logits in enumerate(outputs)]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "id_model.pth")


if __name__ == "__main__":
    train()
