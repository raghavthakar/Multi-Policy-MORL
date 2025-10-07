"""
Replay-memory utilities shared by RL workers and genetic actors.
"""

from __future__ import annotations
import random
from collections import deque
from typing import Deque, List, Tuple, Union

import numpy as np
import torch


class Transition(tuple):
    """
    A single environment step.

    (s, a, r_vec, s', done)  where:
       s, s' : np.ndarray  or torch.Tensor
       a     : int | np.ndarray
       r_vec : np.ndarray  (m objectives)
       done  : bool
    """


class ReplayBuffer:
    """
    Large FIFO buffer for RL workers.  Fixed size, uniform sampling.
    """

    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 action_type: str,
                 action_dim: int,
                 capacity: int,
                 device: torch.device):
        self.capacity = capacity
        self.device = device

        self._storage: Deque[Transition] = deque(maxlen=capacity)
        self.obs_shape = obs_shape
        self.action_type = action_type
        self.action_dim = action_dim

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._storage)

    def push(self,
             state: np.ndarray,
             # Action can be int or array
             action: Union[int, np.ndarray],
             reward_vec: np.ndarray,
             next_state: np.ndarray,
             done: bool) -> None:
        """Store one transition."""
        self._storage.append(
            Transition((state.copy(), action, reward_vec.copy(),
                        next_state.copy(), done)))

    def sample(self, batch_size: int):
        """Return torch tensors on the correct device."""
        batch: List[Transition] = random.sample(self._storage, batch_size)

        states, actions, rewards, next_states, dones = map(
            np.stack, zip(*batch))

        # Convert to torch
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        actions = torch.from_numpy(actions).float().to(self.device)

        return states, actions, rewards, next_states, dones


class MiniBuffer:
    """
    Small per-genome replay used in PDERL / MO-distilled crossover.
    Circular buffer with cheap numpy storage for full transitions.
    """

    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 action_type: str,
                 action_dim: int,
                 max_steps: int = 8_192,):
        self.max_steps = max_steps
        self.action_type = action_type
        self.action_dim = action_dim
        self.ptr = 0
        self.full = False

        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Returns the number of valid transitions currently in the buffer."""
        return self.max_steps if self.full else self.ptr

    def clear(self):
        """Resets the buffer to an empty state."""
        self.ptr = 0
        self.full = False
        # Reset arrays to allow for re-initialization if data shapes change
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def _lazy_init(self, state: np.ndarray, reward_vec: np.ndarray):
        """Initializes storage arrays on first call to add()."""
        if self.states is None:
            obs_shape = state.shape
            num_objectives = reward_vec.shape[0]

            self.states = np.zeros((self.max_steps, *obs_shape), dtype=np.float32)
            self.rewards = np.zeros((self.max_steps, num_objectives), dtype=np.float32)
            self.next_states = np.zeros((self.max_steps, *obs_shape), dtype=np.float32)
            self.dones = np.zeros(self.max_steps, dtype=bool)

            # Initialize actions array
            self.actions = np.zeros((self.max_steps, self.action_dim), dtype=np.float32)


    def add(self, state: np.ndarray, action: Union[int, np.ndarray], reward_vec: np.ndarray, next_state: np.ndarray, done: bool):
        """Adds a new transition to the buffer."""
        self._lazy_init(state, reward_vec)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward_vec
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_steps
        if self.ptr == 0:
            self.full = True

    def add_batch(self, states: np.ndarray, actions: np.ndarray, reward_vecs: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """Adds a batch of transitions to the buffer, handling wraparound."""
        batch_size = len(states)
        if batch_size == 0:
            return

        self._lazy_init(states[0], reward_vecs[0])

        if self.ptr + batch_size >= self.max_steps:
            self.full = True
            num_to_end = self.max_steps - self.ptr
            
            # First part
            self.states[self.ptr:] = states[:num_to_end]
            self.actions[self.ptr:] = actions[:num_to_end]
            self.rewards[self.ptr:] = reward_vecs[:num_to_end]
            self.next_states[self.ptr:] = next_states[:num_to_end]
            self.dones[self.ptr:] = dones[:num_to_end]
            
            # Second part (wraparound)
            remaining = batch_size - num_to_end
            if remaining > 0:
                self.states[:remaining] = states[num_to_end:]
                self.actions[:remaining] = actions[num_to_end:]
                self.rewards[:remaining] = reward_vecs[num_to_end:]
                self.next_states[:remaining] = next_states[num_to_end:]
                self.dones[:remaining] = dones[num_to_end:]
        else:
            indices = np.arange(self.ptr, self.ptr + batch_size)
            self.states[indices] = states
            self.actions[indices] = actions
            self.rewards[indices] = reward_vecs
            self.next_states[indices] = next_states
            self.dones[indices] = dones

        self.ptr = (self.ptr + batch_size) % self.max_steps

    def sample(self, batch_size: int, device: torch.device):
        """Samples a batch of transitions and returns them as torch tensors."""
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        
        idxs = np.random.randint(0, len(self), size=batch_size)

        s = torch.from_numpy(self.states[idxs]).float().to(device)
        r = torch.from_numpy(self.rewards[idxs]).float().to(device)
        s2 = torch.from_numpy(self.next_states[idxs]).float().to(device)
        d = torch.from_numpy(self.dones[idxs].astype(np.uint8)).float().to(device)

        # Conditionally cast actions tensor based on its own dtype
        # This is robust even if the buffer stores mixed types (it shouldn't)
        if self.actions.dtype == np.int64 or self.actions.dtype == np.int32:
            a = torch.from_numpy(self.actions[idxs]).long().to(device)
        else:
            a = torch.from_numpy(self.actions[idxs]).float().to(device)

        return s, a, r, s2, d

    def shuffle(self):
        """In-place random permutation of all *valid* entries."""
        length = len(self)
        if length < 2:
            return

        perm = np.random.permutation(length)
        self.states[:length] = self.states[perm]
        self.actions[:length] = self.actions[perm]
        self.rewards[:length] = self.rewards[perm]
        self.next_states[:length] = self.next_states[perm]
        self.dones[:length] = self.dones[perm]