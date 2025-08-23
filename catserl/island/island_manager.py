# catserl/island/island_manager.py
from __future__ import annotations
from typing import List, Dict
import numpy as np, random, torch
import hashlib
import mo_gymnasium as mo_gym
import gymnasium as gym

from catserl.shared.rl import RLWorker
from catserl.shared.rollout import rollout
from catserl.shared import actors


class IslandManager:
    """
    Warm-up “island” that owns
      - ONE private env instance
      - ONE RLWorker trained on a scalar weight vector
    Public API kept minimal so the global orchestrator can treat all
    objectives uniformly.
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 env,
                 island_id: int,
                 scalar_weight: np.ndarray,
                 cfg: Dict,
                 seed: int,
                 device: torch.device):
        """
        Parameters
        ----------
        env: mo_gymnasium env
            E.g.: run mo_gymnasium.make("mo-mountaincar-v0") to get an env.
        island_id : int
            A unique identifier for the island.
        scalar_weight : np.ndarray
            One-hot (or general) weight vector w_j used to scalarise reward.
        cfg : dict
            Top-level config;.
        seed : int
            Seed for the wrapped env so different ERLManagers get decorrelated
            stochasticity but the run is reproducible.
        device : torch.device
            Where the networks live.
        """
        self.island_id = island_id
        self.cfg = cfg
        self.env = env
        self.rl_alg_name = 'td3'
        # -------------------------------------------------------------- #
        # local deterministic RNG for this island
        self.rs = np.random.RandomState(seed)
        # -------------------------------------------------------------- #
        self.w = scalar_weight.astype(np.float32)

        # Inspect the environment's action space to generalize.
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_type = "discrete"
            self.action_dim = self.env.action_space.n
            self.max_action = None # Cannot be retrieved for discrete gym envs
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.action_type = "continuous"
            self.action_dim = self.env.action_space.shape[0]
            self.max_action = self.env.action_space.high[0]
        else:
            raise ValueError(f"Unsupported action space: {type(self.env.action_space)}")

        self.worker = RLWorker(self.rl_alg_name,
                               self.env.observation_space.shape,
                               self.action_type,
                               self.action_dim,
                               self.max_action,
                               self.w,
                               cfg["rl"],
                               device)

        self.pop = [actors.Actor(self.rl_alg_name,
                                 self.island_id,
                                 self.env.observation_space.shape,
                                 self.action_type,
                                 self.action_dim,
                                 buffer_size=cfg["mini_buffer_size"],
                                 device=device) for _ in range(cfg["pderl"]["pop_size"])]

        self.max_ep_len = cfg["env"].get("max_ep_len", -1)  # default max episode length
        self.gen_counter = 0
        self.migrate_every = int(cfg["pderl"].get("migrate_every_gens", 5))

        # Stats
        self.scalar_returns: List[float] = []
        self.vector_returns: List[np.ndarray] = []
        self.frames_collected = 0

        # Migration log for RL→GA events
        self.migration_log = []

    # ---------- helper: build GA genome from RL policy -----------------
    def _make_rl_actor(self):
        flat, hid = self.worker.export_policy_params()
        rl_actor = actors.Actor(self.rl_alg_name,
                                self.island_id,
                                self.env.observation_space.shape,
                                self.action_type,
                                self.action_dim,
                                hidden_dim=hid,
                                max_action=self.max_action,
                                device=self.worker.device,)
        rl_actor.load_flat_params(flat)
        return rl_actor

    # ------------------------------------------------------------------ #
    # ----------  Warm-up ---------------------------------------------- #
    # ------------------------------------------------------------------ #
    def train(self) -> Dict:
        # --- Training Hyperparameters ---
        total_timesteps = 20000
        start_timesteps = self.worker.agent.rl_kick_in_frames # Get from agent
        update_every_n_steps = 50
        updates_per_session = 50

        # --- Training Loop ---
        state, _ = self.env.reset()
        ep_return_vec = None
        ep_len = 0
        episodes_completed = 0

        for t in range(total_timesteps):
            # Select action: random for the start period, otherwise from the policy
            action = self.worker.act(state)

            # Step the environment
            next_state, reward_vec, done, trunc, _ = self.env.step(action)
            
            # Store the transition in the agent's buffer
            self.worker.remember(state, action, reward_vec, next_state, done)

            # Accumulate episode rewards
            if ep_return_vec is None:
                ep_return_vec = np.array(reward_vec, dtype=np.float32)
            else:
                ep_return_vec += reward_vec

            state = next_state
            ep_len += 1

            # Perform learning updates
            if t >= start_timesteps and t % update_every_n_steps == 0:
                for _ in range(updates_per_session):
                    self.worker.update()

            # Handle episode termination
            if done or trunc:
                # Log episode results
                scalar_return = (ep_return_vec * self.w).sum()
                print(f"Total Steps: {t+1}, Episode: {episodes_completed+1}, Ep. Length: {ep_len}, Scalar Return: {scalar_return:.2f}")
                
                # Store metrics in the manager
                self.scalar_returns.append(scalar_return)
                self.vector_returns.append(ep_return_vec)
                
                # Reset for next episode
                state, _ = self.env.reset()
                ep_return_vec = None
                ep_len = 0
                episodes_completed += 1

    # ------------------------------------------------------------------ #
    # ----------  Accessors needed by later stages  --------------------- #
    # ------------------------------------------------------------------ #
    def critic(self):
        """Return the worker's critic network (used by distilled crossover)."""
        return self.worker.critic()

    def get_scalar_returns(self):
        return self.scalar_returns

    def get_vector_returns(self):
        return self.vector_returns
    
    def export_island(self):
        """
        Export the current state of the island, including the population,
        the RL worker's critic, and the scalarising weight vector.

        Returns
        -------
        pop : list of GeneticActor
            The current population of genetic actors.
        island_id: int
            The identifier for the island.
        critic : torch.nn.Module
            The critic network used by the RL worker.
        w : np.ndarray
            The scalarising weight vector for this island.
        """
        self.pop.append(self._make_rl_actor())
        return self.pop, self.island_id, self.worker.critic(), self.w