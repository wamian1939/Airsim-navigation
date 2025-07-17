from envs.airsim_env import AirSimUAVEnv

env = AirSimUAVEnv()
depth, state = env.reset()

print("Initial state:", state.shape, "Depth shape:", depth.shape)

for _ in range(10):
    action = [1.0, 1.0, 0.0, 0.0]  # 让它往前飞
    (depth, state), reward, done = env.step(action)
    print(f"Step: reward={reward:.3f}, done={done}")
    if done:
        break
