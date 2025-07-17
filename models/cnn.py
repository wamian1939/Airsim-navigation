# models/cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),  # [B, 1, 80, 100] → [B, 8, 80, 100]
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → [B, 8, 40, 50]
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                          # → [B, 16, 20, 25]
            nn.Conv2d(16, 25, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))              # → [B, 25, 1, 1]
        )
        self.output_dim = output_dim

    def forward(self, x):  # x: [B, 1, H, W]
        x = self.conv(x)
        return x.view(-1, self.output_dim)  # → [B, 25]
