from __future__ import annotations
from typing import Tuple, Optional
import copy

import numpy as np
import torch
from catserl.shared.policies import DiscretePolicy
from catserl.shared.buffers import MiniBuffer


class DQNActor:
    """Arg‑max policy with its own buffer."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        hidden_dim: int = 128,
        buffer_size: int = 8192,
        device: str | torch.device = "cpu",
    ) -> None:
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        self.net = DiscretePolicy(obs_shape, n_actions, hidden_dim).to(self.device)
        self.buffer = MiniBuffer(obs_shape, max_steps=buffer_size)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return int(self.net(t).argmax(1).item())

    def remember(self, state, action, r_vec, next_state, done):
        self.buffer.add(state, action, r_vec, next_state, done)

    # ------------------------------------------------------------------
    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.net.parameters()])

    def load_flat_params(self, flat: torch.Tensor) -> None:
        idx = 0
        for p in self.net.parameters():
            n = p.numel()
            p.data.copy_(flat[idx : idx + n].view_as(p))
            idx += n

    def clone(self) -> "DQNActor":
        twin = DQNActor(
            self.obs_shape,
            self.n_actions,
            self.hidden_dim,
            buffer_size=self.buffer.max_steps,
            device=self.device,
        )
        twin.load_flat_params(self.flat_params().clone())
        twin.buffer = copy.deepcopy(self.buffer)
        return twin


class Actor:
    """Generic wrapper that exposes common fields for checkpointing."""

    def __init__(
        self,
        kind: str,
        pop_id: int | None,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        hidden_dim: int = 128,
        buffer_size: int = 8192,
        device: str | torch.device = "cpu",
    ) -> None:
        self.kind = kind.lower()
        self.pop_id = pop_id
        self.fitness: Optional[float] = None
        self.vector_return: Optional[np.ndarray] = None

        if self.kind == "dqn":
            self.impl = DQNActor(
                obs_shape,
                n_actions,
                hidden_dim=hidden_dim,
                buffer_size=buffer_size,
                device=device,
            )
        else:
            raise ValueError(f"unknown actor type: {self.kind}")

        # expose implementation details needed by checkpoint
        self.obs_shape = self.impl.obs_shape
        self.n_actions = self.impl.n_actions
        self.hidden_dim = self.impl.hidden_dim

    # ------------------------------------------------------------------
    def act(self, state):
        return self.impl.act(state)

    def remember(self, *tr):
        self.impl.remember(*tr)

    def flat_params(self):
        return self.impl.flat_params()

    def load_flat_params(self, flat):
        self.impl.load_flat_params(flat)

    def clone(self):
        twin = Actor.__new__(Actor)
        twin.kind = self.kind
        twin.pop_id = self.pop_id
        twin.impl = self.impl.clone()
        return twin
