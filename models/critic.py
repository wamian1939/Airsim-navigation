from models.cnn import CNNEncoder
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.cnn = CNNEncoder(output_dim=25)
        total_input = 25 + state_dim + action_dim

        self.q1 = nn.Sequential(
            nn.Linear(total_input, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(total_input, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, depth, state, action):
        feat = self.cnn(depth)
        x = torch.cat([feat, state, action], dim=1)
        return self.q1(x), self.q2(x)

    def Q1(self, depth, state, action):
        feat = self.cnn(depth)
        x = torch.cat([feat, state, action], dim=1)
        return self.q1(x)
