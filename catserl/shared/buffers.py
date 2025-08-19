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
    Circular buffer with cheap numpy storage for full transitions.
    """

    def __init__(self,
                 obs_shape: Tuple[int, ...],
                 max_steps: int = 8_192,):
        self.max_steps = max_steps
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

        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def add(self, state: np.ndarray, action: int, reward_vec: np.ndarray, next_state: np.ndarray, done: bool):
        """Adds a new transition to the buffer."""
        # --- Lazy Initialization Block ---
        if self.states is None:
            # First time add() is called, so we now know the shapes.
            obs_shape = state.shape
            num_objectives = reward_vec.shape[0]

            # Now, pre-allocate the storage.
            self.states = np.zeros((self.max_steps, *obs_shape), dtype=np.float32)
            self.actions = np.zeros(self.max_steps, dtype=np.int64)
            self.rewards = np.zeros((self.max_steps, num_objectives), dtype=np.float32)
            self.next_states = np.zeros((self.max_steps, *obs_shape), dtype=np.float32)
            self.dones = np.zeros(self.max_steps, dtype=bool)
        # --- End of Block ---

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward_vec
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_steps
        if self.ptr == 0:
            self.full = True
            
    def add_batch(self, states: np.ndarray, actions: np.ndarray, reward_vecs: np.ndarray, next_states: np.ndarray, dones: np.ndarray):
        """
        Adds a batch of transitions to the buffer, handling wraparound.
        """
        batch_size = len(states)
        if batch_size == 0:
            return

        # --- Lazy Initialization Block ---
        if self.states is None:
            obs_shape = states.shape[1:]
            num_objectives = reward_vecs.shape[1]

            self.states = np.zeros((self.max_steps, *obs_shape), dtype=np.float32)
            self.actions = np.zeros(self.max_steps, dtype=np.int64)
            self.rewards = np.zeros((self.max_steps, num_objectives), dtype=np.float32)
            self.next_states = np.zeros((self.max_steps, *obs_shape), dtype=np.float32)
            self.dones = np.zeros(self.max_steps, dtype=bool)
        # --- End of Block ---

        # Check if the batch will wrap around the buffer's capacity
        if self.ptr + batch_size >= self.max_steps:
            # The buffer will be full after this operation
            self.full = True
            
            # Calculate how many transitions fit before the end of the array
            num_to_end = self.max_steps - self.ptr
            
            # First part of the batch: fill until the end
            self.states[self.ptr:] = states[:num_to_end]
            self.actions[self.ptr:] = actions[:num_to_end]
            self.rewards[self.ptr:] = reward_vecs[:num_to_end]
            self.next_states[self.ptr:] = next_states[:num_to_end]
            self.dones[self.ptr:] = dones[:num_to_end]
            
            # Second part of the batch: wrap around and fill from the beginning
            remaining = batch_size - num_to_end
            if remaining > 0:
                self.states[:remaining] = states[num_to_end:]
                self.actions[:remaining] = actions[num_to_end:]
                self.rewards[:remaining] = reward_vecs[num_to_end:]
                self.next_states[:remaining] = next_states[num_to_end:]
                self.dones[:remaining] = dones[num_to_end:]
        else:
            # The batch fits without wrapping around
            indices = np.arange(self.ptr, self.ptr + batch_size)
            self.states[indices] = states
            self.actions[indices] = actions
            self.rewards[indices] = reward_vecs
            self.next_states[indices] = next_states
            self.dones[indices] = dones

        # Update the pointer
        self.ptr = (self.ptr + batch_size) % self.max_steps

    def sample(self, batch_size: int, device: torch.device):
        """Samples a batch of transitions and returns them as torch tensors."""
        idxs = np.random.randint(0, len(self), size=batch_size)

        s = torch.from_numpy(self.states[idxs]).float().to(device)
        a = torch.from_numpy(self.actions[idxs]).long().to(device)
        r = torch.from_numpy(self.rewards[idxs]).float().to(device)
        s2 = torch.from_numpy(self.next_states[idxs]).float().to(device)
        d = torch.from_numpy(self.dones[idxs].astype(np.uint8)).float().to(device)

        return s, a, r, s2, d
    
    def shuffle(self):
        """
        In-place random permutation of all *valid* entries.
        """
        length = len(self)
        if length < 2:
            return

        perm = np.random.permutation(length)
        self.states[:length] = self.states[perm]
        self.actions[:length] = self.actions[perm]
        self.rewards[:length] = self.rewards[perm]
        self.next_states[:length] = self.next_states[perm]
        self.dones[:length] = self.dones[perm]
