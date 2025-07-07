# catserl/erl_manager.py
from __future__ import annotations
from typing import List, Dict
import numpy as np
import torch

from catserl.envs.four_room import FourRoomWrapper
from catserl.rl.dqn import RLWorker
from catserl.envs.rollout import rollout


class ERLManager:
    """
    Warm-up “island” that owns
      • ONE private env instance
      • ONE RLWorker trained on a scalar weight vector
      • (later) a GA sub-population and PDERL operators

    Public API kept minimal so the global orchestrator can treat all
    objectives uniformly.
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 scalar_weight: np.ndarray,
                 cfg: Dict,
                 seed: int,
                 device: torch.device):
        """
        Parameters
        ----------
        scalar_weight : np.ndarray
            One-hot (or general) weight vector w_j used to scalarise reward.
        cfg : dict
            Top-level config; expects sub-dict ``cfg["dqn"]``.
        seed : int
            Seed for the wrapped env so different ERLManagers get decorrelated
            stochasticity but the run is reproducible.
        device : torch.device
            Where the networks live.
        """
        self.env = FourRoomWrapper(seed=seed, beta=cfg["env"]["beta_novelty"])
        self.w = scalar_weight.astype(np.float32)

        self.worker = RLWorker(self.env.observation_space.shape,
                               self.env.action_space.n,
                               self.w,
                               cfg["dqn"],
                               device)

        # Stats
        self.scalar_returns: List[float] = []
        self.vector_returns: List[np.ndarray] = []
        self.frames_collected = 0

    # ------------------------------------------------------------------ #
    # ----------  Warm-up generation loop  ------------------------------ #
    # ------------------------------------------------------------------ #
    def train_generation(self, episodes: int = 1) -> Dict:
        """
        Collect `episodes` rollouts, let the RL worker learn online.

        Returns
        -------
        Dict with simple metrics you can aggregate or log:
            {
              "mean_scalar_return": ... ,
              "episodes": episodes,
              "frames": int,
            }
        """
        for _ in range(episodes):
            ret_vec, ep_len, ext_ret_vec = rollout(self.env, self.worker, learn=True)
            ret_scalar = float((ret_vec * self.w).sum())

            self.vector_returns.append(ret_vec)
            self.scalar_returns.append(ret_scalar)
            self.frames_collected += ep_len

        return dict(mean_scalar_return=np.mean(self.scalar_returns[-episodes:]),
                    episodes=episodes,
                    frames=self.frames_collected,
                    ext_ret_vec=ext_ret_vec,)

    # ------------------------------------------------------------------ #
    # ----------  Accessors needed by later stages  --------------------- #
    # ------------------------------------------------------------------ #
    def critic(self):
        """Return the worker's critic network (used by distilled crossover)."""
        return self.worker.critic()

    def actor_state_dict(self):
        """Flat copy of the current policy weights (for RL → GA sync)."""
        return self.worker.actor_state_dict()

    def get_scalar_returns(self):
        return self.scalar_returns

    def get_vector_returns(self):
        return self.vector_returns
