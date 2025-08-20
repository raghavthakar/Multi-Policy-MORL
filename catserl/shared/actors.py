# actors.py
from __future__ import annotations
from typing import Tuple, Optional, Union
import copy

import numpy as np
import torch
from catserl.shared.policies import DiscretePolicy
from catserl.shared.buffers import MiniBuffer


class _DQNActorImpl:
    """DQN-specific implementation: a policy network and a replay buffer."""

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_type: str,
        action_dim: int,
        hidden_dim: int,
        buffer_size: int,
        device: torch.device,
    ) -> None:
        if action_type != "discrete":
            raise ValueError("DQNActorImpl only supports 'discrete' action spaces.")

        self.obs_shape = obs_shape
        self.action_type = action_type
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.net = DiscretePolicy(obs_shape, action_dim, hidden_dim).to(self.device)
        self.buffer = MiniBuffer(
            obs_shape,
            action_type,
            action_dim,
            max_steps=buffer_size
        )

    @torch.no_grad()
    def act(self, state: np.ndarray) -> int:
        """Returns the best action according to the policy network."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        return int(self.net(state_tensor).argmax(1).item())

    def remember(self, state, action, r_vec, next_state, done):
        """Adds a transition to the replay buffer."""
        self.buffer.add(state, action, r_vec, next_state, done)

    def flat_params(self) -> torch.Tensor:
        """Flattens all network parameters into a single tensor."""
        return torch.cat([p.data.view(-1) for p in self.net.parameters()])

    def load_flat_params(self, flat: torch.Tensor) -> None:
        """Loads flattened parameters into the network."""
        idx = 0
        for p in self.net.parameters():
            n = p.numel()
            p.data.copy_(flat[idx : idx + n].view_as(p))
            idx += n


class Actor:
    """A generic actor that wraps a specific algorithm implementation."""

    def __init__(
        self,
        kind: str,
        pop_id: int | None,
        obs_shape: Tuple[int, ...],
        action_type: str,
        action_dim: int,
        hidden_dim: int = 128,
        buffer_size: int = 8192,
        device: str | torch.device = "cpu",
    ) -> None:
        self.kind = kind.lower()
        self.pop_id = pop_id
        self.fitness: Optional[float] = None
        self.vector_return: Optional[np.ndarray] = None
        
        self.action_type = action_type
        self.action_dim = action_dim

        if self.kind == "dqn":
            self._impl = _DQNActorImpl(
                obs_shape,
                action_type,
                action_dim,
                hidden_dim=hidden_dim,
                buffer_size=buffer_size,
                device=torch.device(device),
            )
        else:
            raise ValueError(f"Unknown actor kind: {self.kind}")

    # --- Properties to provide clean access to implementation details ---
    @property
    def policy(self) -> DiscretePolicy:
        return self._impl.net

    @property
    def buffer(self) -> MiniBuffer:
        return self._impl.buffer

    @buffer.setter
    def buffer(self, value: MiniBuffer) -> None:
        self._impl.buffer = value

    @property
    def obs_shape(self) -> Tuple[int, ...]:
        return self._impl.obs_shape

    @property
    def n_actions(self) -> int:
        # Keep n_actions for discrete-specific cases, but raise error if used improperly
        if self.action_type != "discrete":
            raise AttributeError("'.n_actions' is only available for discrete action spaces.")
        return self.action_dim

    @property
    def hidden_dim(self) -> int:
        return self._impl.hidden_dim

    # --- Delegated methods ---
    def act(self, state: np.ndarray) -> Union[int, np.ndarray]:
        return self._impl.act(state)

    def remember(self, *transition):
        self._impl.remember(*transition)

    def flat_params(self) -> torch.Tensor:
        return self._impl.flat_params()

    def load_flat_params(self, flat: torch.Tensor):
        self._impl.load_flat_params(flat)

    def clone(self) -> "Actor":
        """Creates a new actor with the same configuration and network weights."""
        twin = Actor(
            kind=self.kind,
            pop_id=None,
            obs_shape=self.obs_shape,
            action_type=self.action_type,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            buffer_size=self.buffer.max_steps,
            device=self._impl.device,
        )
        twin.load_flat_params(self.flat_params())
        return twin