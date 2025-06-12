import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model import IDRecognizer
from utils import preprocess_image, transform
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDRecognizer().to(device)
    dataset = IDDataset("data/train", "data/train_labels.txt")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epoch_losses = []

    for epoch in range(16):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)  # list of logits

            loss = sum(
                [criterion(logits, labels[:, i]) for i, logits in enumerate(outputs)]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "id_model.pth")

    with open("loss_log.txt", "w") as f:
        f.write("Epoch\tLoss\n")
        for epoch, loss in enumerate(epoch_losses, 1):
            f.write(f"{epoch}\t{loss:.6f}\n")

    # 绘制 loss 曲线
    # plt.figure()
    # plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")


if __name__ == "__main__":
    train()
