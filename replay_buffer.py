import numpy as np
import random

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(5e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        # ✅ 新增 depth 图缓存区（默认大小：80x100）
        self.depth = np.zeros((max_size, 1, 256, 144), dtype=np.float32)
        self.next_depth = np.zeros((max_size, 1, 256, 144), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, depth, next_depth):
        i = self.ptr
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = reward
        self.next_state[i] = next_state
        self.done[i] = float(done)

        self.depth[i] = depth    # shape: [1, 80, 100]
        self.next_depth[i] = next_depth

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind],
            self.depth[ind],
            self.next_depth[ind],
        )
