import torch.nn as nn


class IDRecognizer(nn.Module):
    def __init__(self, num_digits=14, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25x100
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x50
        )
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64 * 12 * 5, 128), nn.ReLU(), nn.Linear(128, num_classes)
                )
                for _ in range(num_digits)
            ]
        )

    def forward(self, x):
        x = self.features(x)  # [B, 64, 12, 5]
        x = x.view(x.size(0), -1)  # [B, 64*12*5]
        out = [classifier(x) for classifier in self.classifiers]
        return out  # list of [B, num_classes]
