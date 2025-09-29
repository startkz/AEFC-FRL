import gym
from .cps_env import CPSEnvWrapper

def make_envs(env_id: str, n: int, max_steps: int = 200, reward_flip: bool = False):
    envs = []
    for _ in range(n):
        env = gym.make(env_id)
        envs.append(CPSEnvWrapper(env, max_steps=max_steps, reward_flip=reward_flip))
    return envs
