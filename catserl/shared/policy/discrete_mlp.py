# policy/discrete_mlp.py

"""
Feed-forward networks for discrete‐action domains.
* DiscreteActor : π(s) logits  (GeneticActor genome)
* DuelingQNet   : V(s) + A(s,a) critic  (DQN worker)
"""

import numpy as np
import torch
import torch.nn as nn


def _init_linear(layer: nn.Linear) -> None:
    # Kaiming uniform for small MLPs keeps initial Q small
    nn.init.kaiming_uniform_(layer.weight, a=0.0)
    nn.init.zeros_(layer.bias)


class DiscreteActor(nn.Module):
    def __init__(self, obs_shape: tuple[int,...], n_actions: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                _init_linear(m)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [B,|A|]


class DuelingQNet(nn.Module):
    def __init__(self,
                 obs_shape: tuple[int,...],
                 n_actions: int,
                 hidden_dim: int = 128):
        super().__init__()
        in_dim = int(np.prod(obs_shape))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim), nn.ReLU()
        )
        self.v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        for net in (self.shared, self.v, self.a):
            for m in net:
                if isinstance(m, nn.Linear):
                    _init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.v(h)                # [B,1]
        a = self.a(h)                # [B,|A|]
        q = v + a - a.mean(dim=1, keepdim=True)
        return q                    # [B,|A|]
