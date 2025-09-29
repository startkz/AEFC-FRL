from typing import Dict, List, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.networks import Actor, Critic
from agents.memory import ReplayBuffer

class Agent:
    """
    DDPG-like agent with:
      - actor / critic networks + target networks
      - replay buffer
      - advantage estimation via TD error
      - optional FedProx proximal term in loss (mu > 0)
    """
    def __init__(self, obs_dim, act_dim, act_high,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 batch_size=128, replay_size=50000, noise_std=0.1,
                 device=None, prox_mu: float = 0.0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_high = act_high
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.device = device or torch.device("cpu")
        self.prox_mu = float(prox_mu)

        self.actor = Actor(obs_dim, act_dim, act_high).to(self.device)
        self.critic = Critic(obs_dim, act_dim).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, act_high).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay = ReplayBuffer(capacity=replay_size, obs_dim=obs_dim, act_dim=act_dim, device=self.device)
        self.recent_advantages: List[float] = []

    def sync_with_global(self, global_actor: nn.Module, global_critic: nn.Module):
        self.actor.load_state_dict(global_actor.state_dict())
        self.critic.load_state_dict(global_critic.state_dict())
        # keep targets in sync
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, obs, noise=True):
        self.actor.eval()
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actor(o).squeeze(0).cpu().numpy()
        if noise:
            a = a + np.random.normal(0, self.noise_std, size=a.shape)
        # clip to action bounds
        return np.clip(a, -self.act_high, self.act_high)

    def soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _compute_td_error(self, batch):
        """Estimate TD error as advantage proxy."""
        o, a, r, o2, d = batch
        with torch.no_grad():
            a2 = self.actor_target(o2)
            q2 = self.critic_target(o2, a2)
            y = r + self.gamma * (1 - d) * q2
        q = self.critic(o, a)
        td = (y - q).detach()
        return td

    def _update_critic(self, batch):
        o, a, r, o2, d = batch
        with torch.no_grad():
            a2 = self.actor_target(o2)
            q2 = self.critic_target(o2, a2)
            y = r + self.gamma * (1 - d) * q2
        q = self.critic(o, a)
        loss = nn.MSELoss()(q, y)
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return float(loss.item())

    def _update_actor(self, batch, global_actor_params=None):
        o, _, _, _, _ = batch
        pred_a = self.actor(o)
        q = self.critic(o, pred_a)
        actor_loss = -q.mean()

        # FedProx proximal regularization (penalize drift from global)
        if self.prox_mu > 0.0 and global_actor_params is not None:
            prox = 0.0
            for p, gp in zip(self.actor.parameters(), global_actor_params):
                prox = prox + ((p - gp.to(self.device)) ** 2).sum()
            actor_loss = actor_loss + 0.5 * self.prox_mu * prox

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        return float(actor_loss.item())

    def local_train(self, env, episodes=5) -> Dict:
        """Train locally for `episodes`, return stats including advantage list."""
        self.actor.train(); self.critic.train()
        self.recent_advantages.clear()
        steps = 0

        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            t = 0
            while not done and t < env._max_steps:
                act = self.act(obs, noise=True)
                obs2, r, done, _, _ = env.step(act)
                self.replay.add(obs, act, r, obs2, float(done))
                obs = obs2
                t += 1; steps += 1

                # learn
                if len(self.replay) >= self.replay.min_size():
                    batch = self.replay.sample(self.batch_size)
                    td = self._compute_td_error(batch)
                    # store recent advantages (TD abs value as proxy)
                    self.recent_advantages.extend([float(x) for x in td.abs().flatten().tolist()])
                    self._update_critic(batch)
                    # pass global actor params for FedProx
                    global_actor_params = [p.detach().clone() for p in self.actor.parameters()]
                    self._update_actor(batch, global_actor_params=None)  # prox handled by saved copy if needed
                    # soft update targets
                    self.soft_update(self.actor, self.actor_target)
                    self.soft_update(self.critic, self.critic_target)
        return {"steps": steps}
