"""
Replay-memory utilities shared by RL workers and genetic actors.
"""

from __future__ import annotations
import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch


class Transition(tuple):
    """
    A single environment step.

    (s, a, r_vec, s', done)  where:
       s, s' : np.ndarray  or torch.Tensor
       a     : int
       r_vec : np.ndarray  (m objectives)
       done  : bool
    """


class ReplayBuffer:
    """
    Large FIFO buffer for RL workers (DQN).  Fixed size, uniform sampling.
    """

    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 capacity: int,
                 device: torch.device):
        self.capacity = capacity
        self.device = device

        self._storage: Deque[Transition] = deque(maxlen=capacity)
        self.obs_shape = obs_shape

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._storage)

    def push(self,
             state: np.ndarray,
             action: int,
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
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones


class MiniBuffer:
    """
    Small per-genome replay used in PDERL / MO-distilled crossover.
    Circular buffer with cheap numpy storage.
    """

    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 max_steps: int = 8_192):
        self.max_steps = max_steps
        self.ptr = 0
        self.full = False

        # pre-allocate
        self.states = np.zeros((max_steps, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(max_steps, dtype=np.int64)

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.max_steps if self.full else self.ptr

    def add(self, state: np.ndarray, action: int):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action

        self.ptr = (self.ptr + 1) % self.max_steps
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, len(self), size=batch_size)
        s = torch.from_numpy(self.states[idxs]).float().to(device)
        a = torch.from_numpy(self.actions[idxs]).long().to(device)
        return s, a
    
    # ------------------------------------------------------------------ #
    def shuffle(self):
        """
        In-place random permutation of all *valid* entries.

        • Does nothing if the buffer holds < 2 samples.  
        • Leaves `ptr` and `full` unchanged, so chronology resumes correctly
          for any new transitions appended after the shuffle.
        """
        length = len(self)           # valid slice: 0 … length-1
        if length < 2:
            return

        perm = np.random.permutation(length)
        # When full, length == self.max_steps; otherwise == self.ptr
        self.states[:length]  = self.states[perm]
        self.actions[:length] = self.actions[perm]
