import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.airsim_env import AirSimUAVEnv
from models.actor import Actor
from models.critic import Critic
from replay_buffer import ReplayBuffer
from td3_trainer import TD3Trainer

# === 超参数 ===
state_dim = 8
action_dim = 4
max_action = 15.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化环境、模型、经验池 ===
env = AirSimUAVEnv()
actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim).to(device)
trainer = TD3Trainer(state_dim, action_dim, max_action, actor, critic)
replay_buffer = ReplayBuffer(state_dim, action_dim)

# === 训练配置 ===
total_episodes = 50
max_steps = 1000
start_training_after = 500
train_every = 2
all_rewards = []

# === 训练主循环 ===
for ep in tqdm(range(total_episodes), desc="Training"):
    depth, state = env.reset()
    episode_reward = 0

    inner_bar = tqdm(range(max_steps), desc=f"Ep {ep+1}", leave=False)

    for step in inner_bar:
        # ✅ reshape depth to [1, 1, H, W]
        def preprocess_depth(d): return d.reshape(1, 1, 256, 144)
        depth_tensor = torch.FloatTensor(preprocess_depth(depth)).to(device)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        if replay_buffer.size < start_training_after:
            action = np.random.uniform(-max_action, max_action, size=action_dim)
        else:
            actor.eval()
            with torch.no_grad():
                action = actor(depth_tensor, state_tensor).cpu().numpy().flatten()
            action += np.random.normal(0, 0.5, size=action_dim)
            action = np.clip(action, -max_action, max_action)

        (next_depth, next_state), reward, done = env.step(action)

        # 存经验
        replay_buffer.add(state, action, reward, next_state, done,
                          preprocess_depth(depth), preprocess_depth(next_depth))

        state, depth = next_state, next_depth
        episode_reward += reward

        inner_bar.set_postfix(
            step_reward=f"{reward:.2f}",
            total_reward=f"{episode_reward:.1f}",
            buffer=replay_buffer.size
        )

        # 训练
        if replay_buffer.size >= start_training_after:
            for _ in range(train_every):
                trainer.train(replay_buffer, batch_size=128)

        if done:
            break

    all_rewards.append(episode_reward)
    print(f"🎯 Ep {ep+1}: Reward={episode_reward:.2f}  Buffer={replay_buffer.size}")

    # ✅ 保存轨迹图
    traj = env.get_trajectory()
    xs, ys, zs = zip(*traj)
    plt.figure()
    plt.plot(xs, ys, 'b-o', markersize=2, label="Path")
    plt.scatter([env.goal_pos[0]], [env.goal_pos[1]], c='green', s=80, label="Goal")
    plt.scatter([xs[0]], [ys[0]], c='red', s=80, label="Start")
    plt.axis('equal')
    plt.legend()
    plt.title(f"Trajectory - Ep {ep+1}")
    plt.tight_layout()
    plt.savefig(f"trajectory_ep{ep+1}.png")
    plt.close()

# ✅ 总奖励曲线
plt.figure()
plt.plot(all_rewards, label="Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("TD3 Training Reward Curve")
plt.legend()
plt.grid()
plt.savefig("reward_curve.png")
print("✅ Saved reward_curve.png")
