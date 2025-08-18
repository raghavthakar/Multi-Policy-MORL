"""
GeneticActor
============

A genome =  (policy  +  per‑actor replay buffer  +  bookkeeping fields)

* Provides `.act`, `.remember`, `.clone`, and flat‑parameter helpers.
* Stores runtime stats (`fitness`, `vector_return`) so that evaluation and
  selection code can read/write without separate dicts.
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch, numpy as np
import copy
import torch.nn.functional as F

from catserl.shared.policy.discrete_mlp import DiscreteActor
from catserl.shared.data.buffers import MiniBuffer


class GeneticActor:
    # ------------------------------------------------------------------ #
    def __init__(self,
                 pop_id: int | None,
                 obs_shape: Tuple[int, ...],
                 n_actions: int,
                 hidden_dim: int = 128,
                 buffer_size: int = 8_192,
                 device: str | torch.device = "cpu"):
        # immutable meta
        self.pop_id = pop_id
        self.obs_shape  = obs_shape
        self.n_actions  = n_actions
        self.hidden_dim = hidden_dim
        self.device     = torch.device(device)

        # policy network
        self.net = DiscreteActor(obs_shape, n_actions, hidden_dim).to(self.device)

        # tiny per‑genome replay buffer (<s,a>)
        self.buffer = MiniBuffer(obs_shape, max_steps=buffer_size)

        # runtime evaluation stats (updated by eval_pop or island_step)
        self.fitness: Optional[float]        = None     # scalar
        self.vector_return: Optional[np.ndarray] = None # full vector

    # ------------------------------------------------------------------ #
    # Acting (deterministic policy)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        """
        Deterministic action = argmax(logits).
        """
        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits = self.net(t)
        probabilities = F.softmax(logits, dim=1)
        action = torch.multinomial(probabilities, num_samples=1).item()
        return action

    # ------------------------------------------------------------------ #
    # remember  (used by rollout to populate MiniBuffer)
    # ------------------------------------------------------------------ #
    def remember(self, state, action, r_vec, next_state, done):
        """
        rollout() calls policy.remember(s, a, r_vec, s2, done).
        """
        self.buffer.add(state, action, r_vec, next_state, done)

    # ------------------------------------------------------------------ #
    # Flat‑parameter helpers  (for mutation & cloning)
    # ------------------------------------------------------------------ #
    def flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.net.parameters()])

    def load_flat_params(self, flat: torch.Tensor) -> None:
        """Load from a 1‑D tensor obtained via .flat_params()."""
        idx = 0
        for p in self.net.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].view_as(p))
            idx += n

    # ------------------------------------------------------------------ #
    # Deep clone (new buffer, copied weights)
    # ------------------------------------------------------------------ #
    def clone(self) -> "GeneticActor":
        """Creates a deep clone with a copied buffer and network weights."""
        clone = GeneticActor(self.pop_id,
                             self.obs_shape,
                             self.n_actions,
                             self.hidden_dim,
                             buffer_size=self.buffer.max_steps,
                             device=self.device)
        clone.load_flat_params(self.flat_params().clone())
        
        # --- Simplified buffer cloning ---
        clone.buffer = copy.deepcopy(self.buffer)

        return clone
