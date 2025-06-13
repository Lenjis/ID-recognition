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
    dataset = IDDataset("data/train", "data/train_labels.txt")  # 训练数据集
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True
    )  # 使用 DataLoader 载入数据

    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器

    epoch_losses = []
    epoch_losses_test = []

    for epoch in range(40):
        model.train()
        running_loss = 0.0

        for imgs, labels in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)  # list of logits

            loss = torch.stack(
                [
                    criterion(logits, labels[:, i]) for i, logits in enumerate(outputs)
                ]  # i = 0, 1, ..., 13
            ).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(dataset)  # 计算每次迭代平均损失
        epoch_losses.append(epoch_loss)
        torch.save(model.state_dict(), "id_model.pth")

        test_loss = infer("data/test")
        epoch_losses_test.append(test_loss)
        print(
            f"Epoch {epoch}, Loss_train: {epoch_loss:.4f}, Loss_test: {test_loss:.4f}"
        )

    with open("loss_log.txt", "w") as f:
        f.write("Epoch\tLoss_train\tLoss_test\n")
        for epoch, loss in enumerate(epoch_losses, 1):
            f.write(f"{epoch}\t{loss:.6f}\t{epoch_losses_test[epoch-1]:.6f}\n")


def infer(test_path, INFER_MODE="test"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IDRecognizer().to(device)
    model.load_state_dict(torch.load("id_model.pth", map_location=device))
    model.eval()

    testset = IDDataset(test_path, f"data/{INFER_MODE}_labels.txt")
    dataloader = DataLoader(testset, batch_size=1, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    losses = 0.0
    total_correct = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = torch.stack(
            [criterion(logits, labels[:, i]) for i, logits in enumerate(outputs)]
        ).sum()
        losses += loss.item()
        predictions = [torch.argmax(logits, dim=1) for logits in outputs]

        # print(f"Predictions: {"".join(str(p.item()) for p in predictions)}")
        # print(f"True:        {"".join(str(c.item()) for c in labels[0])}")

        correct = sum(
            [p.item() == labels[0, i].item() for i, p in enumerate(predictions)]
        )
        total_correct += correct
    accuracy = total_correct / (len(testset) * 14)  # 14位数字
    print(f"Accuracy: {accuracy*100:.4f}%")

    avg_loss = losses / len(testset)  # 计算平均损失
    # print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    train()
