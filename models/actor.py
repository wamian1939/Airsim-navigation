# models/actor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import CNNEncoder

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.cnn = CNNEncoder(output_dim=25)
        self.fc = nn.Sequential(
            nn.Linear(25 + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, depth, state):  # ✅ 必须接受两个输入
        feat = self.cnn(depth)
        x = torch.cat([feat, state], dim=1)
        return self.fc(x) * self.max_action
