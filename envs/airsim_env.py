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

        self.action_duration = 0.4
        self.max_episode_steps = 1000
        self.goal_radius = 2.0
        self.step_count = 0
        self.trajectory = []

        self.stuck_history = []
        self.stuck_threshold = 10
        self.min_altitude = -1.0
        self.max_stuck_attempts = 3

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        self.client.moveToZAsync(-10, velocity=2).join()

        # ✅ 每个 episode 随机目标点
        theta = random.uniform(0, 2 * np.pi)
        r = 35
        self.goal_pos = np.array([r * np.cos(theta), r * np.sin(theta), -10.0])
        self.prev_distance = self._compute_distance_to_goal()

        self.step_count = 0
        self.trajectory = []
        self.stuck_history = []
        self.client.simFlushPersistentMarkers()

        return self._get_obs()

    def step(self, action):
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        vx, vy, vz, yaw_rate = [float(a) for a in action]
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=self.action_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()
        self.client.moveToZAsync(-10, velocity=5).join()

        next_obs = self._get_obs()

        # 判定卡住（可选）
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        v = state.kinematics_estimated.linear_velocity
        speed = np.linalg.norm([v.x_val, v.y_val, v.z_val])
        dist_to_goal = np.linalg.norm(
            np.array([pos.x_val, pos.y_val, pos.z_val]) - self.goal_pos
        )

        if speed < 0.1 and abs(pos.z_val + 10) < 1.0 and dist_to_goal > self.goal_radius:
            print(f"🛑 Detected hover/stuck (speed={speed:.2f}, z={pos.z_val:.2f}), ending episode.")
            return next_obs, -50.0, True

        reward, done = self._compute_reward()
        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            done = True

        if done:
            self.client.simPlotPoints(
                self.trajectory, color_rgba=[1, 0, 0, 1], size=10.0, is_persistent=True
            )
            self.client.simPlotPoints(
                [airsim.Vector3r(*self.goal_pos)], color_rgba=[0, 1, 0, 1], size=30.0, is_persistent=True
            )

        return next_obs, reward, done

    def _get_obs(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True)
        ])
        depth = np.array(responses[0].image_data_float, dtype=np.float32)

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        obs_vector = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            0.0, 0.0
        ])

        self.trajectory.append(airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val))
        return depth, obs_vector

    def _compute_reward(self):
        pos = self._get_position()
        dist_to_goal = np.linalg.norm(pos - self.goal_pos)

        reward = 0.0
        done = False

        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 100.0
        else:
            vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
            speed = np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
            reward += 0.1 * speed
            reward += 1.0 / (dist_to_goal + 1.0)

            if pos[2] > self.min_altitude:
                reward += -5.0 * (pos[2] - self.min_altitude)

        return reward, done

    def _get_position(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def _compute_distance_to_goal(self):
        return np.linalg.norm(self._get_position() - self.goal_pos)

    def get_trajectory(self):
        return [(p.x_val, p.y_val, p.z_val) for p in self.trajectory]
