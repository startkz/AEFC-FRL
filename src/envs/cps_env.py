import gym
import numpy as np

class CPSEnvWrapper(gym.Wrapper):
    """
    Gym wrapper to enforce max steps and optional reward flipping attack.
    Used for Pendulum-v1 to emulate CPS-like stabilization metric.
    """
    def __init__(self, env, max_steps=200, reward_flip=False):
        super().__init__(env)
        self._max_steps = max_steps
        self._steps = 0
        self._reward_flip = reward_flip

    def reset(self, **kwargs):
        self._steps = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        self._steps += 1
        obs, reward, done, truncated, info = self.env.step(action)
        if self._reward_flip:
            reward = -reward
        if self._steps >= self._max_steps:
            done = True
        return obs, reward, done, truncated, info
