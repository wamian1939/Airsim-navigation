# envs/airsim_env.py

import airsim
import numpy as np
import time
import random

class AirSimUAVEnv:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.action_duration = 0.5
        self.max_episode_steps = 500
        self.goal_radius = 2.0
        self.step_count = 0

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        self.client.moveToZAsync(-10, velocity=2).join()

        # 🎯 设置目标点：65米半径圆周上随机选点
        theta = random.uniform(0, 2 * np.pi)
        r = 65
        self.goal_pos = np.array([r * np.cos(theta), r * np.sin(theta), 5.0])
        self.prev_distance = self._compute_distance_to_goal()

        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        vx, vy, vz, yaw_rate = action

        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=self.action_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()

        self.client.moveToZAsync(-10, velocity=5).join()

        next_obs = self._get_obs()
        reward, done = self._compute_reward()
        self.step_count += 1

        if self.step_count >= self.max_episode_steps:
            done = True

        return next_obs, reward, done

    def _get_obs(self):
        # 🎥 获取深度图
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True)
        ])
        depth = np.array(responses[0].image_data_float, dtype=np.float32)

        # 🧭 获取自状态
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        obs_vector = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            0.0,  # yaw_err
            0.0   # yaw_rate
        ])

        return depth, obs_vector

    def _compute_reward(self):
        current_pos = self._get_position()
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        re = (self.prev_distance - curr_dist) / 65.0  # normalize

        reward = 5.0 * re
        done = False

        # 成功到达
        if curr_dist < 2:
            reward += 10
            done = True

        self.prev_distance = curr_dist
        return reward, done

    def _get_position(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def _compute_distance_to_goal(self):
        return np.linalg.norm(self._get_position() - self.goal_pos)
