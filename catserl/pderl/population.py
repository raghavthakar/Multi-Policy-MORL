"""
GeneticActor: a single genome = policy network + mini-replay buffer.
No mutation/crossover yet; those will arrive in safe_mutation.py etc.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np, torch
from catserl.policy.discrete_mlp import DiscreteActor
from catserl.data.buffers import MiniBuffer


class GeneticActor:
    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 n_actions: int,
                 hidden_dim: int = 128,
                 buffer_size: int = 8_192,
                 device="cpu"):
        self.obs_shape   = obs_shape          # ← store originals
        self.n_actions   = n_actions
        self.hidden_dim  = hidden_dim
        self.device      = torch.device(device)

        self.net = DiscreteActor(obs_shape, n_actions, hidden_dim).to(self.device)
        self.buffer = MiniBuffer(obs_shape, max_steps=buffer_size)

    # ---------- acting -------------------------------------------------- #
    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return int(self.net(t).argmax(dim=1).item())
    
    # ---------- store transitions so crossover / mutation can sample ---- #
    def remember(self, state, action, *unused):
        """
        Signature compatible with rollout’s call:
            remember(s, a, r_vec, s2, done)

        We only need (state, action) for the MiniBuffer; the rest is ignored.
        """
        self.buffer.add(state, action)

    # ---------- flat parameter helpers --------------------------------- #
    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.net.parameters()])

    def load_flat_params(self, flat: torch.Tensor):
        idx = 0
        for p in self.net.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].view_as(p))
            idx += n

    # ---------- cloning (used later by mutation) ------------------------ #
    def clone(self) -> "GeneticActor":
        clone = GeneticActor(self.obs_shape,
                             self.n_actions,
                             self.hidden_dim,
                             buffer_size=len(self.buffer.states),
                             device=self.device)
        clone.load_flat_params(self.flat_params().clone())
        return clone
