import torch
import torch.nn as nn
import copy

def mlp(sizes, act=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [act()]
        elif out_act is not None:
            layers += [out_act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_high):
        super().__init__()
        self.net = mlp([obs_dim, 128, 128, act_dim])
        self.act_high = act_high
        self.tanh = nn.Tanh()

    def forward(self, obs):
        x = self.net(obs)
        # scale to action bounds
        return self.tanh(x) * self.act_high

    def clone(self):
        new = Actor(self.net[0].in_features, self.net[-1].out_features, self.act_high)
        new.load_state_dict(copy.deepcopy(self.state_dict()))
        return new

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, 128, 128, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)
