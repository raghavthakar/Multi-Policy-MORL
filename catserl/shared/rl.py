# catserl/shared/rl.py
from __future__ import annotations
"""Minimal pluggable‑algorithm RL worker

This module keeps **all** algorithm logic in one file while allowing you to
swap RL algorithms by passing a string (e.g. "dqn", "ppo") to ``RLWorker``.

The only abstraction is a tiny *duck‑typed* ``Algo`` base class that defines
five methods.  Each concrete algorithm class (``DQN`` below) implements those
methods so the worker can delegate to it.

Add new algorithms by:
1. Creating another class that follows the same interface.
2. Adding an ``elif`` branch in ``RLWorker.__init__``.

No global registry or separate packages are required.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# -----------------------------------------------------------------------------
# Tiny *informal* interface – every algorithm must expose these five methods.
# -----------------------------------------------------------------------------


class Algo:  # noqa: D101 – doctring would just repeat the method names
    """Lightweight base class documenting the required public surface.

    We do **not** use ABC / abstractmethod to avoid an extra import and keep
    the dependency graph flat – algorithms simply override the methods they
    need.  If a method is left unimplemented we want Python to raise a
    ``NotImplementedError`` at runtime, which mirrors an ABC anyway.
    """

    # Each concrete class should provide explicit type hints to satisfy mypy.
    def act(self, state: np.ndarray) -> int:  # noqa: D401 – imperative mood
        raise NotImplementedError

    def remember(self, *transition) -> None:  # noqa: D401 – imperative mood
        raise NotImplementedError

    def update(self) -> None:  # noqa: D401 – imperative mood
        raise NotImplementedError

    def save(self, path: str) -> None:  # noqa: D401 – imperative mood
        raise NotImplementedError

    def load(self, path: str) -> None:  # noqa: D401 – imperative mood
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Dueling‑DQN implementation (identical learning logic as the original worker)
# -----------------------------------------------------------------------------


from catserl.shared.buffers import ReplayBuffer  # isort: skip  (project import)


def _init_linear(layer: nn.Linear) -> None:
    """Kaiming‑uniform initialization keeps early Q‑values small and stable."""

    nn.init.kaiming_uniform_(layer.weight, a=0.0)
    nn.init.zeros_(layer.bias)


class DuelingQNet(nn.Module):
    """2‑stream Dueling architecture with shared torso (simple MLP)."""

    def __init__(
        self, obs_shape: Tuple[int, ...], n_actions: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        in_dim = int(np.prod(obs_shape))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Initialize every Linear layer
        for net in (self.shared, self.v, self.a):
            for m in net:
                if isinstance(m, nn.Linear):
                    _init_linear(m)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return per‑action Q‑values with aggregating **advantage** + **value**.

        Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
        """

        h = self.shared(x)
        v = self.v(h)  # [B, 1]
        a = self.a(h)  # [B, |A|]
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """State‑value helper for monitoring (no action dimension)."""

        h = self.shared(x)
        v = self.v(h).squeeze(-1)
        return v


class DQN(Algo):
    """Dueling‑DQN + Double‑DQN updates + soft target network.

    This class contains **all** learning components: networks, replay buffer,
    optimizer, ε‑schedule and update rules.  RLWorker will treat it as a black
    box implementing the small `Algo` interface.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_type: str,
        action_dim: int,
        scalar_weight: np.ndarray,
        cfg: dict,
        device: torch.device,
    ) -> None:
        if action_type != "discrete":
            raise ValueError("DQN algorithm only supports discrete action spaces.")
        
        self.device = device
        self.n_actions = action_dim
        self.scalar_weight = torch.tensor(scalar_weight, device=device)

        # Networks ----------------------------------------------------
        hid = int(cfg.get("hidden_dim", 128))
        self.net = DuelingQNet(obs_shape, self.n_actions, hid).to(device)
        self.tgt = DuelingQNet(obs_shape, self.n_actions, hid).to(device)
        self.tgt.load_state_dict(self.net.state_dict())

        # Optimizer & hyper‑parameters --------------------------------
        self.lr = float(cfg.get("lr", 1e-4))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))  # soft‑update coeff.
        self.optim = Adam(self.net.parameters(), lr=self.lr)

        # Replay buffer ----------------------------------------------
        self.buffer = ReplayBuffer(
            obs_shape,
            action_type,
            action_dim,
            capacity=int(cfg.get("buffer_size", 100_000)),
            device=device,
        )

        # ε‑greedy schedule ------------------------------------------
        self.eps_start = float(cfg.get("eps_start", 1.0))
        self.eps_end = float(cfg.get("eps_end", 0.05))
        self.eps_decay = int(cfg.get("eps_decay_frames", 50_000))
        self.frame_idx = 0  # increments every env‑step

        # Update cadence ---------------------------------------------
        self.batch_size = int(cfg.get("batch_size", 64))
        self.update_every = int(cfg.get("update_every", 4))
        self.update_ctr = 0

    # ------------------------------------------------------------------
    # Public API (satisfies Algo interface)
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:
        """ε‑greedy action selection using the online network."""

        self.frame_idx += 1
        if np.random.rand() < self._epsilon():
            return np.random.randint(self.n_actions)

        t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.net(t)  # [1, |A|]
        return int(q.argmax(dim=1).item())

    def remember(self, *transition) -> None:
        """Push (s, a, r_vec, s2, done) into replay buffer."""

        self.buffer.push(*transition)

    def update(self) -> None:
        """One Double‑DQN update if enough data is available."""

        if len(self.buffer) < self.batch_size:
            return

        # ------------------------------------------------------------------
        # Sample batch & compute targets
        # ------------------------------------------------------------------
        s, a, r_vec, s2, d = self.buffer.sample(self.batch_size)
        d = d.unsqueeze(1)  # [B, 1]
        # Scalarize multi‑objective reward
        r = (r_vec * self.scalar_weight).sum(dim=1, keepdim=True)  # [B, 1]

        # Current Q(s,a)
        q = self.net(s).gather(1, a.unsqueeze(1))

        with torch.no_grad():
            # Double‑DQN: action from *online* net, value from *target* net
            best_a = self.net(s2).argmax(dim=1, keepdim=True)
            q_next = self.tgt(s2).gather(1, best_a)
            y = r + self.gamma * (1 - d) * q_next  # TD target

        # ------------------------------------------------------------------
        # Optimize
        # ------------------------------------------------------------------
        loss = F.smooth_l1_loss(q, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Soft‑update target parameters
        for p, tp in zip(self.net.parameters(), self.tgt.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def save(self, path: str) -> None:
        torch.save(
            {
                "net": self.net.state_dict(),
                "tgt": self.tgt.state_dict(),
                "opt": self.optim.state_dict(),
                "frame": self.frame_idx,
            },
            path,
        )

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.tgt.load_state_dict(ck["tgt"])
        self.optim.load_state_dict(ck["opt"])
        self.frame_idx = ck["frame"]

    # ------------------------------------------------------------------
    # Extras (not part of Algo but handy for GA / diagnostics)
    # ------------------------------------------------------------------
    def export_policy_params(self):  # noqa: D401 – imperative mood
        """Flatten weights that affect arg‑max policy for downstream GA."""

        with torch.no_grad():
            flat = torch.cat(
                [
                    self.net.shared[1].weight.flatten(),
                    self.net.shared[1].bias,
                    self.net.a[0].weight.flatten(),
                    self.net.a[0].bias,
                    self.net.a[2].weight.flatten(),
                    self.net.a[2].bias,
                ]
            ).cpu().clone()

        hidden_dim = self.net.shared[1].out_features
        return flat, hidden_dim

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _epsilon(self) -> float:
        """Linearly decay ε over ``eps_decay`` frames."""

        frac = min(1.0, self.frame_idx / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)


# -----------------------------------------------------------------------------
# RLWorker – orchestration shell that delegates to a chosen algorithm class
# -----------------------------------------------------------------------------


class RLWorker:
    """Environment‑agnostic worker that can run multiple RL algorithms.

    The worker owns a **single** concrete algorithm instance.  Delegation keeps
    public signatures identical to the original design so existing training
    loops require zero changes beyond specifying the ``algo`` string.
    """

    def __init__(
        self,
        algo: str,
        obs_shape: Tuple[int, ...],
        action_type: str,
        action_dim: int,
        scalar_weight: np.ndarray,
        cfg: dict,
        device: torch.device,
    ) -> None:
        algo = algo.lower()
        if algo == "dqn":
            self.agent: Algo = DQN(
                obs_shape, action_type, action_dim, scalar_weight, cfg, device
            )
        else:
            raise ValueError(
                f"Unknown algorithm '{algo}'. Add an elif branch in RLWorker.__init__."
            )
        
        self.device = device

    # ------------------------------------------------------------------
    # Thin delegation layer – forwards calls to the underlying algorithm.
    # ------------------------------------------------------------------
    def act(self, state: np.ndarray) -> int:  # noqa: D401 – imperative mood
        return self.agent.act(state)

    def remember(self, *transition) -> None:  # noqa: D401 – imperative mood
        self.agent.remember(*transition)

    def update(self) -> None:  # noqa: D401 – imperative mood
        self.agent.update()

    def save(self, path: str) -> None:  # noqa: D401 – imperative mood
        self.agent.save(path)

    def load(self, path: str) -> None:  # noqa: D401 – imperative mood
        self.agent.load(path)

    def export_policy_params(self):  # noqa: D401 – imperative mood
        return getattr(self.agent, "export_policy_params", None)()
    
    def critic(self):
        return self.agent.net