# main.py

import numpy as np
import torch
from envs.airsim_env import AirSimUAVEnv
from models.actor import Actor
from models.critic import Critic
from replay_buffer import ReplayBuffer
from td3_trainer import TD3Trainer
import matplotlib.pyplot as plt


# 初始化环境
env = AirSimUAVEnv()
state_dim = 8
action_dim = 4
max_action = 10.0

all_rewards = []


# 初始化网络 & Trainer
actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
trainer = TD3Trainer(state_dim, action_dim, max_action, actor, critic)

# 初始化经验池
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=50000)

# 训练参数
total_episodes = 1000
max_steps = 500
start_training_after = 10000  # 收集多少经验后开始训练
train_every = 5               # 每步训练几次

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for episode in range(total_episodes):
    print(f"\n🚁 Episode {episode + 1}")
    depth, state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 动作选择（有探索噪声）
        if replay_buffer.size < start_training_after:
            action = np.random.uniform(-max_action, max_action, size=action_dim)
            action[2] = 0.0
        else:
            actor.eval()
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            action += np.random.normal(0, 0.5, size=action_dim)  # 探索噪声
            action = np.clip(action, -max_action, max_action)
            action[2] = 0.0

        (next_depth, next_state), reward, done = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        all_rewards.append(episode_reward)


        # 开始训练
        if replay_buffer.size >= start_training_after:
            for _ in range(train_every):
                trainer.train(replay_buffer, batch_size=128)

        if done:
            break

    

    print(f"🎯 Reward: {episode_reward:.2f} | Steps: {step + 1} | Buffer: {replay_buffer.size}")
    
# ✅ 所有 episodes 训练完之后再画一次图
plt.figure()
plt.plot(all_rewards, label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("TD3 Training Reward Curve")
plt.legend()
plt.grid()
plt.savefig("reward_curve.png")
print("✅ Reward curve saved to reward_curve.png")
