from typing import Dict
import numpy as np
import torch

def evaluate_policy(actor, env, episodes=10) -> Dict:
    """Run deterministic evaluation of the actor on a single env."""
    actor = actor.to("cpu")
    actor.eval()
    rewards = []
    with torch.no_grad():
        for _ in range(episodes):
            obs, _ = env.reset()
            ep_r = 0.0
            done = False
            steps = 0
            while not done and steps < env._max_steps:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                act = actor(obs_t).squeeze(0).numpy()
                obs, r, done, _, _ = env.step(act)
                ep_r += float(r)
                steps += 1
            rewards.append(ep_r)
    avg_r = float(np.mean(rewards))
    # Map reward to 0..100 "accuracy" scale for presentation
    # For Pendulum, reward in [-2000, 0] roughly; normalize
    acc = max(0.0, min(100.0, 100.0 * (avg_r + 2000.0) / 2000.0))
    return {"avg_reward": avg_r, "accuracy_pct": acc}
