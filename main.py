import numpy as np
import torch
from envs.airsim_env import AirSimUAVEnv
from models.actor import Actor
from models.critic import Critic
from replay_buffer import ReplayBuffer
from td3_trainer import TD3Trainer
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 初始化环境
env = AirSimUAVEnv()
state_dim = 8
action_dim = 4
max_action = 15.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络 & Trainer
actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim).to(device)
trainer = TD3Trainer(state_dim, action_dim, max_action, actor, critic)

# 初始化经验池
replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=50000)

# 训练参数
total_episodes = 50
max_steps = 1000
start_training_after = 500
train_every = 2

# 保存每个 episode 的总奖励
all_rewards = []

# 训练主循环
for episode in tqdm(range(total_episodes), desc="Training Progress"):
    print(f"\n🚁 Episode {episode + 1}")
    depth, state = env.reset()
    episode_reward = 0
    progress_bar = tqdm(range(max_steps), desc=f"Episode {episode + 1}/{total_episodes}", leave=False)

    for step in progress_bar:
        # ✅ reshape depth → [1, 1, 256, 144]
        depth = depth.reshape(256, 144)[np.newaxis, :, :]
        depth_tensor = torch.FloatTensor(depth).unsqueeze(0).to(device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # 动作选择
        if replay_buffer.size < start_training_after:
            action = np.random.uniform(-max_action, max_action, size=action_dim)
            action[2] = 0.0  # 锁死 vz
        else:
            actor.eval()
            with torch.no_grad():
                action = actor(depth_tensor, state_tensor).cpu().numpy().flatten()
            action += np.random.normal(0, 0.5, size=action_dim)
            action = np.clip(action, -max_action, max_action)
            action[2] = 0.0  # 只在平面移动

        # 与环境交互
        (next_depth, next_state), reward, done = env.step(action)

        # ✅ reshape 当前和下一个 depth 为 [1, H, W]
        depth = depth.reshape(256, 144)[np.newaxis, :, :]
        next_depth = next_depth.reshape(256, 144)[np.newaxis, :, :]

        # 存入经验池
        replay_buffer.add(state, action, reward, next_state, done, depth, next_depth)

        # 更新状态
        state = next_state
        depth = next_depth
        episode_reward += reward

        progress_bar.set_postfix(reward=f"{reward:.2f}", total=f"{episode_reward:.2f}")

        # 开始训练
        if replay_buffer.size >= start_training_after:
            for _ in range(train_every):
                trainer.train(replay_buffer, batch_size=128)

        if done:
            break


    all_rewards.append(episode_reward)
    print(f"🎯 Reward: {episode_reward:.2f} | Steps: {step + 1} | Buffer: {replay_buffer.size}")
    print("depth shape:", depth.shape)
    trajectory = env.get_trajectory()
    xs, ys, zs = zip(*trajectory)
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o', markersize=2, label="UAV path", color='blue')
    plt.scatter([env.goal_pos[0]], [env.goal_pos[1]], color='green', s=80, label="Goal")
    plt.scatter([xs[0]], [ys[0]], color='red', s=80, label="Start")
    plt.title(f"Trajectory - Episode {episode + 1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"trajectory_ep{episode + 1}.png")
    plt.close()
    print(f"🖼️ Trajectory plot saved to: trajectory_ep{episode + 1}.png")




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
