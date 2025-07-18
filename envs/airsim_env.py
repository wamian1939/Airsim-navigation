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

        # 🔧 新增：卡住检测机制
        self.stuck_history = []  # 记录最近几步的位置和速度
        self.stuck_threshold = 10  # 连续5步检测卡住
        self.min_altitude = -1.0  # 最小飞行高度
        self.max_stuck_attempts = 3  # 最大重新起飞尝试次数

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        time.sleep(0.5)
        self.client.moveToZAsync(-10, velocity=2).join()

        # 🎯 设置目标点：65 米半径圆周随机位置
        theta = random.uniform(0, 2 * np.pi)
        r = 35
        self.goal_pos = np.array([r * np.cos(theta), r * np.sin(theta), -10.0])
        self.prev_distance = self._compute_distance_to_goal()

        # ✅ 清空轨迹和卡住历史
        self.step_count = 0
        self.trajectory = []
        self.stuck_history = []
        self.client.simFlushPersistentMarkers()

        return self._get_obs()

    def step(self, action):
        # 1. 拿回控制 & 解锁
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 2. 执行动作
        vx, vy, vz, yaw_rate = [float(a) for a in action]
        self.client.moveByVelocityAsync(
            vx, vy, vz,
            duration=self.action_duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        ).join()
        self.client.moveToZAsync(-10, velocity=5).join()

        # 3. 取新观测
        next_obs = self._get_obs()

        # 4. 检测是否“安全悬停”导致落地不动
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        v = state.kinematics_estimated.linear_velocity
        speed = np.linalg.norm([v.x_val, v.y_val, v.z_val])
        dist_to_goal = np.linalg.norm(
            np.array([pos.x_val, pos.y_val, pos.z_val]) - self.goal_pos
        )

        # 如果速度接近 0、与平台高度相当，且还没到目标 → 直接当作 done
        if speed < 0.1 and abs(pos.z_val + 10) < 1.0 and dist_to_goal > self.goal_radius:
            # 这里给一个较大的负奖励，迫使 agent 不要把自己卡住
            stuck_reward = -50
            print(f"🛑 Detected hover/stuck (speed={speed:.2f}, z={pos.z_val:.2f}), ending episode.")
            return next_obs, stuck_reward, True

        # 5. 正常计算 reward 和 done
        reward, done = self._compute_reward()
        self.step_count += 1

        # 6. Episode 超时或到达目标时绘图并结束
        if self.step_count >= self.max_episode_steps or done:
            self.client.simPlotPoints(
                self.trajectory, color_rgba=[1,0,0,1], size=10.0, is_persistent=True
            )
            self.client.simPlotPoints(
                [airsim.Vector3r(*self.goal_pos)], color_rgba=[0,1,0,1], size=30.0, is_persistent=True
            )
            done = True

        return next_obs, reward, done

    def _is_stuck(self):
        """🔧 改进的卡住检测机制"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        current_state = {
            'position': np.array([pos.x_val, pos.y_val, pos.z_val]),
            'velocity': np.array([vel.x_val, vel.y_val, vel.z_val]),
            'altitude': pos.z_val,
            'speed': np.linalg.norm([vel.x_val, vel.y_val, vel.z_val])
        }

        # 添加到历史记录
        self.stuck_history.append(current_state)
        if len(self.stuck_history) > self.stuck_threshold:
            self.stuck_history.pop(0)

        # 需要足够的历史数据
        if len(self.stuck_history) < self.stuck_threshold:
            return False

        # 检测条件1：高度过低
        if current_state['altitude'] > self.min_altitude:
            return True

        # 检测条件2：速度持续过低
        recent_speeds = [s['speed'] for s in self.stuck_history]
        if all(speed < 0.3 for speed in recent_speeds):
            return True

        # 检测条件3：位置几乎没有变化
        recent_positions = [s['position'] for s in self.stuck_history]
        position_changes = []
        for i in range(1, len(recent_positions)):
            change = np.linalg.norm(recent_positions[i] - recent_positions[i - 1])
            position_changes.append(change)

        if all(change < 0.5 for change in position_changes):
            return True

        return False

    def _attempt_recovery(self):
        """🔧 飞机卡住后的恢复机制"""
        for attempt in range(self.max_stuck_attempts):
            print(f"🔄 Recovery attempt {attempt + 1}/{self.max_stuck_attempts}")

            try:
                # Step 1: 重新启用控制
                self.client.enableApiControl(True)
                self.client.armDisarm(True)
                time.sleep(0.2)

                # Step 2: 强制设置到安全位置
                current_pos = self.client.getMultirotorState().kinematics_estimated.position
                safe_pose = airsim.Pose(
                    airsim.Vector3r(current_pos.x_val, current_pos.y_val, -10),
                    airsim.to_quaternion(0, 0, 0)
                )
                self.client.simSetVehiclePose(safe_pose, ignore_collision=True)
                time.sleep(0.3)

                # Step 3: 重新起飞
                self.client.takeoffAsync().join()
                time.sleep(0.5)

                # Step 4: 确保到达目标高度
                self.client.moveToZAsync(-10, velocity=3).join()
                time.sleep(0.3)

                # Step 5: 唤醒 UAV（确保螺旋桨转动）
                print("✨ Sending small move command to exit hover...")
                self.client.moveByVelocityAsync(0.1, 0.0, 0.0, duration=0.5).join()

                # Step 6: 控制状态确认
                print(f"🧪 isApiControlEnabled = {self.client.isApiControlEnabled()}")

                # Step 5: 验证恢复是否成功
                new_state = self.client.getMultirotorState()
                new_pos = new_state.kinematics_estimated.position
                new_vel = new_state.kinematics_estimated.linear_velocity
                new_speed = np.linalg.norm([new_vel.x_val, new_vel.y_val, new_vel.z_val])

                print(f"📍 After recovery: altitude={new_pos.z_val:.2f}, speed={new_speed:.2f}")

                # 验证恢复成功的条件
                if new_pos.z_val < -7 and abs(new_pos.z_val - (-10)) < 2:
                    print("✅ Recovery successful!")
                    self.stuck_history = []  # 清空历史记录
                    return True

            except Exception as e:
                print(f"❌ Recovery attempt {attempt + 1} failed: {e}")
                continue

        print("❌ All recovery attempts failed")
        return False

    def _get_obs(self):
        # 🎥 获取深度图
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True)
        ])
        depth = np.array(responses[0].image_data_float, dtype=np.float32)

        # 🧭 获取状态
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity

        obs_vector = np.array([
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            0.0,  # yaw_err
            0.0  # yaw_rate
        ])

        # ✅ 每步记录 UAV 位置
        self.trajectory.append(airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val))

        return depth, obs_vector

    def _compute_reward(self):
        current_pos = self._get_position()
        curr_dist = np.linalg.norm(current_pos - self.goal_pos)
        progress_reward = (self.prev_distance - curr_dist) / 35.0
        distance_penalty = -curr_dist / 35.0

        reward = 20.0 * progress_reward + 2.0 * distance_penalty

        # 🔧 新增：高度惩罚（防止飞机飞得太低）
        altitude_penalty = 0
        if current_pos[2] > self.min_altitude:
            altitude_penalty = -5.0 * (current_pos[2] - self.min_altitude)

        reward += altitude_penalty
        done = False

        # ❌ 碰撞惩罚
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            reward -= 10

        # ✅ 成功到达目标点
        if curr_dist < self.goal_radius:
            reward += 10000
            done = True

        self.prev_distance = curr_dist
        return reward, done

    def _get_position(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def _compute_distance_to_goal(self):
        return np.linalg.norm(self._get_position() - self.goal_pos)

    def get_trajectory(self):
        # 返回当前 episode 的 UAV 路径坐标 [(x1,y1,z1), ...]
        return [(p.x_val, p.y_val, p.z_val) for p in self.trajectory]