import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class TD3Trainer:
    def __init__(self, state_dim, action_dim, max_action, actor, critic):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)

        self.critic = critic
        self.critic_target = copy.deepcopy(critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, replay_buffer, batch_size=128):
        self.total_it += 1

        (
            state, action, reward, next_state, done,
            depth, next_depth
        ) = replay_buffer.sample(batch_size)

        # 转为 tensor
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        depth = torch.FloatTensor(depth).to(self.device)           # [B, 1, H, W]
        next_depth = torch.FloatTensor(next_depth).to(self.device)

        # 生成下一个动作（加噪声）
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_depth, next_state) + noise).clamp(-self.max_action, self.max_action)

            # 目标 Q 值
            target_Q1, target_Q2 = self.critic_target(next_depth, next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * 0.99 * target_Q

        # 当前 Q 值
        current_Q1, current_Q2 = self.critic(depth, state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 延迟更新 Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(depth, state, self.actor(depth, state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新 target 网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)
