import random
import numpy as np
import torch

class ReplayBuffer:
    """Simple uniform replay buffer; TD error used only for advantage logging."""
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def __len__(self): return self.size

    def min_size(self): return min(1000, self.capacity // 10)

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.obs2[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        o = torch.tensor(self.obs[idx], dtype=torch.float32, device=self.device)
        a = torch.tensor(self.act[idx], dtype=torch.float32, device=self.device)
        r = torch.tensor(self.rew[idx], dtype=torch.float32, device=self.device)
        o2 = torch.tensor(self.obs2[idx], dtype=torch.float32, device=self.device)
        d = torch.tensor(self.done[idx], dtype=torch.float32, device=self.device)
        return o, a, r, o2, d
