import torch
import torch.nn as nn

class OCRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(320 * 320 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = OCRNet()