# rl/dqn.py

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from catserl.shared.data.buffers import ReplayBuffer
from catserl.shared.policy.discrete_mlp import DuelingQNet


class RLWorker:
    """
    One Dueling‐DQN worker per objective.
    """

    def __init__(self,
                 obs_shape: Tuple[int,...],
                 n_actions: int,
                 scalar_weight: np.ndarray,
                 cfg: dict,
                 device: torch.device):
        self.device = device
        self.n_actions = n_actions
        self.scalar_weight = torch.tensor(scalar_weight, device=device)

        # networks
        hid = int(cfg.get("hidden_dim", 128))
        self.net = DuelingQNet(obs_shape, n_actions, hidden_dim=hid).to(device)
        self.tgt = DuelingQNet(obs_shape, n_actions, hidden_dim=hid).to(device)
        self.tgt.load_state_dict(self.net.state_dict())

        # optim & hyperparams
        self.lr    = float(cfg.get("lr", 1e-4))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau   = float(cfg.get("tau",   0.005))
        self.optim = Adam(self.net.parameters(), lr=self.lr)

        # replay
        self.buffer = ReplayBuffer(obs_shape,
                                   capacity=int(cfg.get("buffer_size",100_000)),
                                   device=device)

        # ε‐schedule
        self.eps_start = float(cfg.get("eps_start", 1.0))
        self.eps_end   = float(cfg.get("eps_end",   0.05))
        self.eps_decay = int(cfg.get("eps_decay_frames", 50_000))
        self.frame_idx = 0

        # update cadence
        self.batch_size   = int(cfg.get("batch_size", 64))
        self.update_every = int(cfg.get("update_every", 4))
        self.update_ctr   = 0

    def _epsilon(self) -> float:
        frac = min(1.0, self.frame_idx / self.eps_decay)
        return self.eps_start + frac*(self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        self.frame_idx += 1
        if np.random.rand() < self._epsilon():
            return np.random.randint(self.n_actions)
        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.net(t)  # [1, A]
        return int(q.argmax(dim=1).item())

    def remember(self, *tr):
        self.buffer.push(*tr)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        self.update_ctr = (self.update_ctr+1) % self.update_every
        if self.update_ctr != 0:
            return

        s, a, r_vec, s2, d = self.buffer.sample(self.batch_size)
        d = d.unsqueeze(1)                      # [B,1]
        r = (r_vec * self.scalar_weight).sum(dim=1,keepdim=True)

        q = self.net(s).gather(1, a.unsqueeze(1))  # [B,1]

        with torch.no_grad():
            # Double DQN target
            # pick best action under online net
            best_a = self.net(s2).argmax(dim=1, keepdim=True)      # [B,1]
            # evaluate it under the target net
            qn = self.tgt(s2).gather(1, best_a)                    # [B,1]
            y  = r + self.gamma*(1-d)*qn

        loss = F.mse_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # soft‐update
        for p, tp in zip(self.net.parameters(), self.tgt.parameters()):
            tp.data.mul_(1-self.tau).add_(self.tau*p.data)

    def critic(self) -> DuelingQNet:
        return self.net

    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "tgt": self.tgt.state_dict(),
            "opt": self.optim.state_dict(),
            "frame": self.frame_idx,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.tgt.load_state_dict(ck["tgt"])
        self.optim.load_state_dict(ck["opt"])
        self.frame_idx = ck["frame"]

    # ------------------------------------------------------------------
    #  Export current deterministic policy (arg-max over Q) as flat weights
    # ------------------------------------------------------------------
    def export_policy_params(self):
        """
        Returns
        -------
        flat_params : torch.Tensor  (1-D, detached, on CPU)
            Weights that reproduce  π(s)=argmax_a Q(s,a).
        hidden_dim  : int
            Width of the hidden layers (needed by the GA side).
        """
        # Extract weights that matter for logits = Advantage stream
        with torch.no_grad():
            flat = torch.cat([
                self.net.shared[1].weight.flatten(),
                self.net.shared[1].bias,
                self.net.a[0].weight.flatten(),
                self.net.a[0].bias,
                self.net.a[2].weight.flatten(),
                self.net.a[2].bias
            ]).cpu().clone()

        hidden_dim = self.net.shared[1].out_features
        return flat, hidden_dim
