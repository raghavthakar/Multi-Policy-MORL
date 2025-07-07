"""
Dueling-DQN worker used for each objective during the warm-up stage.
Publishes:
    • RLWorker   - handles epsilon-greedy acting, TD updates, target sync
    • DuelingQNet - online / target networks
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from catserl.data.buffers import ReplayBuffer


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class DuelingQNet(nn.Module):
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 n_actions: int,
                 hidden: int = 128):
        super().__init__()
        in_dim = int(np.prod(obs_shape))

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        # value and advantage streams
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    def forward(self, x):
        x = self.shared(x)
        v = self.value(x)                 # [B, 1]
        a = self.advantage(x)             # [B, A]
        q = v + a - a.mean(dim=1, keepdim=True)
        return q


# --------------------------------------------------------------------------- #
# RL worker
# --------------------------------------------------------------------------- #
class RLWorker:
    """
    One worker per objective.  Trains Dueling-DQN on scalarised reward
    (dot product of reward-vector and its one-hot scalar_weight).
    """

    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 n_actions: int,
                 scalar_weight: np.ndarray,          # e_j  (one-hot)
                 cfg: dict,
                 device: torch.device):

        self.device = device
        self.scalar_weight = torch.tensor(scalar_weight,
                                          dtype=torch.float32,
                                          device=device)

        # networks
        self.net = DuelingQNet(obs_shape, n_actions,
                               hidden=cfg.get('hidden_dim', 128)).to(device)
        self.tgt = DuelingQNet(obs_shape, n_actions,
                               hidden=cfg.get('hidden_dim', 128)).to(device)
        self.tgt.load_state_dict(self.net.state_dict())

        self.optim = Adam(self.net.parameters(), lr=float(cfg.get('lr', 1e-4)))
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.005)

        # experience buffer
        self.buffer = ReplayBuffer(obs_shape,
                                   capacity=cfg.get('buffer_size', 100_000),
                                   device=device)

        # epsilon schedule
        self.eps_init = cfg.get('eps_start', 1.0)
        self.eps_final = cfg.get('eps_end', 0.05)
        self.eps_decay = cfg.get('eps_decay_frames', 200_000)
        self.frame_idx = 0

        # misc
        self.batch_size = cfg.get('batch_size', 64)
        self.update_every = cfg.get('update_every', 4)
        self.update_counter = 0

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #
    def _epsilon(self):
        frac = min(1.0, self.frame_idx / self.eps_decay)
        return self.eps_init + frac * (self.eps_final - self.eps_init)

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        """
        ε-greedy action for data collection.
        """
        self.frame_idx += 1
        if np.random.rand() < self._epsilon():
            return np.random.randint(self.net.advantage[-1].out_features)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.net(state_t)            # [1, A]
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------ #
    # learning
    # ------------------------------------------------------------------ #
    def remember(self, *transition):
        self.buffer.push(*transition)

    def update(self):
        """
        Called every env step by outer loop.  Runs a TD-update every
        `update_every` steps if enough data.
        """
        if len(self.buffer) < self.batch_size:
            return

        self.update_counter += 1
        if self.update_counter % self.update_every:
            return

        s, a, r_vec, s2, d = self.buffer.sample(self.batch_size)
        d = d.unsqueeze(1)        # keepdim=True right away

        # scalarise reward: r = w · r_vec
        r = (r_vec * self.scalar_weight).sum(dim=1, keepdim=True)

        q = self.net(s).gather(1, a.unsqueeze(1))            # [B, 1]

        with torch.no_grad():
            q_next = self.tgt(s2).max(dim=1, keepdim=True)[0]
            y = r + self.gamma * (1 - d) * q_next

        loss = F.mse_loss(q, y)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # soft-update target
        with torch.no_grad():
            for p_online, p_tgt in zip(self.net.parameters(),
                                       self.tgt.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(self.tau * p_online.data)

    # ------------------------------------------------------------------ #
    # convenience hooks for evolutionary layer
    # ------------------------------------------------------------------ #
    def critic(self) -> DuelingQNet:
        """Return the online Q-network (required by crossover)."""
        return self.net

    def actor_state_dict(self):
        """
        Extract the part of the Q-net needed to behave as a deterministic
        policy (shared + advantage heads).  Suitable for RL→GA sync.
        """
        sd = self.net.state_dict()
        return {k: v.clone() for k, v in sd.items()
                if k.startswith('shared') or k.startswith('advantage')}

    # ------------------------------------------------------------------ #
    def save(self, path: str):
        torch.save({
            'net': self.net.state_dict(),
            'tgt': self.tgt.state_dict(),
            'optim': self.optim.state_dict(),
            'frame_idx': self.frame_idx,
            'buffer_len': len(self.buffer),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt['net'])
        self.tgt.load_state_dict(ckpt['tgt'])
        self.optim.load_state_dict(ckpt['optim'])
        self.frame_idx = ckpt['frame_idx']
