# catserl/island/island_manager.py
from __future__ import annotations
from typing import List, Dict
import numpy as np, torch
from collections import deque
import mo_gymnasium as mo_gym
import gymnasium as gym

from catserl.shared.rl import RLWorker
from catserl.shared.rollout import deterministic_rollout
from catserl.shared import actors
from catserl.shared.evo_utils import eval_pop
from catserl.shared import checkpoint


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
                 other_scalar_weights: List[np.ndarray],
                 cfg: Dict,
                 checkpointer: checkpoint.Checkpoint = None,
                 seed: int = 2024,
                 device: torch.device = 'cpu'):
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
        self.alg_name = 'pderl'
        self.action_type = 'continuous'
        # -------------------------------------------------------------- #
        self.seed = seed
        # -------------------------------------------------------------- #
        self.w = scalar_weight.astype(np.float32)
        self.other_ws = other_scalar_weights

        # Inspect the environment's action space to generalize.
        if isinstance(self.env.action_space, gym.spaces.Box):
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
                               self.other_ws,
                               cfg["rl"],
                               device)

        self.trained_timesteps = 0
        self.max_ep_len = cfg["env"].get("max_ep_len", -1)  # default max episode length
        self.pop = []

        # Stats
        self.scalar_returns: deque = deque(maxlen=100)
        self.vector_returns: deque = deque(maxlen=100)

        # Checkpointing variables
        self.timesteps_between_checkpoints = 50000
        self.checkpointer = checkpointer

        # Track training variables
        self._training_stats = self.TrainingStats()

        # TD3-specific parameters
        self.update_every_n_steps = cfg['rl']['update_every_n_steps'] # aka training frequency
        self.updates_per_session = cfg['rl']['updates_per_session'] # aka training epochs
    
        # PDERL-specific parameters
        if self.alg_name == 'pderl':
            self.pop = [actors.Actor(kind='td3',
                                     pop_id=0,
                                     obs_shape=self.env.observation_space.shape, 
                                     action_type=self.action_type, 
                                     action_dim=self.action_dim, 
                                     hidden_dim=256, 
                                     max_action=self.max_action, 
                                     device=device) for _ in range(cfg['pderl']['pop_size'])]

    class TrainingStats:
        "Persistent training variables that allow control to pop in and out of the train() method."
        def __init__(self):
            self.state = None
            self.ep_return_vec = None
            self.ep_len = 0
            self.episodes_completed = 0
            self.random_action = True
        
        def get(self):
            return (
                self.state,
                self.ep_return_vec,
                self.ep_len,
                self.episodes_completed,
                self.random_action,
            )
        
        def set(self, state, ep_return_vec, ep_len, episodes_completed, random_action):
            self.state = state
            self.ep_return_vec = ep_return_vec
            self.ep_len = ep_len
            self.episodes_completed = episodes_completed
            self.random_action = random_action

    def resume_from_checkpoint(self) -> None:
        """
        Restores the island's complete training state from the latest snapshot.
        
        This includes the agent's networks, optimizers, the replay buffer,
        and the manager's training progress counters.
        """
        if not self.checkpointer:
            print(f"[Island {self.island_id}] No checkpointer provided, cannot resume.")
            return

        # The load method restores the agent and buffer in-place and
        # returns the manager's state dictionary.
        manager_state = self.checkpointer.load_latest_island_snapshot(
            island_id=self.island_id,
            agent=self.worker.agent,
            buffer=self.worker.buffer()
        )

        if manager_state:
            # manager_state expected to be a dict containing trained_timesteps and training_stats
            self.trained_timesteps = manager_state.get('trained_timesteps', self.trained_timesteps)
            # Attempt to restore training_stats object if present
            if 'training_stats' in manager_state:
                try:
                    self._training_stats = manager_state['training_stats']
                except Exception:
                    # If direct assignment fails, ignore and continue
                    pass
            print(f"[Island {self.island_id}] Resumed training from timestep {self.trained_timesteps}.")
        else:
            print(f"[Island {self.island_id}] No checkpoint found, starting from scratch.")
    
    # ---------- get a deterministic evaluation of the policy -----------
    def _eval_policy(self, episodes_per_actor=10):
        vec_return = np.zeros_like(self.w, dtype=np.float32)
        eval_env = mo_gym.make(self.cfg['env']['name'])
        for ep_num in range(episodes_per_actor):
            ret_vec, ep_len = deterministic_rollout(
                eval_env, self.worker, store_transitions=False, max_ep_len=self.max_ep_len, other_actor=None, seed=self.seed+ep_num
            )  # NOTE: Set learn=True to update actor's buffer
            vec_return += ret_vec

        vec_return = vec_return / episodes_per_actor
        print("Evlauated returns: ", vec_return)

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
    def train(self, steps_to_train=1000) -> Dict:
        # Use a start-up period to populate the buffer with random actions.
        start_timesteps = self.worker.agent.rl_kick_in_frames # Get from agent

        # Initialize environment and training state on the very first step.
        if self.trained_timesteps == 0:
            state, _ = self.env.reset(seed=self.seed)
            self._training_stats.set(state, None, 0, 0, True)

        # --- Training Loop ---
        for _ in range(steps_to_train):
            state, ep_return_vec, ep_len, episodes_completed, random_action = self._training_stats.get()
            
            # Use random actions for the start-up period, otherwise use the policy.
            if self.trained_timesteps < start_timesteps:
                action = self.worker.act(state, random_action=True)
            else:
                action = self.worker.act(state, noisy_action=True)

            # Step the environment
            next_state, reward_vec, done, trunc, _ = self.env.step(action)
            
            # Store the transition in the agent's buffer
            self.worker.remember(state, action, np.array(reward_vec, dtype=np.float32), next_state, done or trunc)

            # Accumulate episode rewards
            if ep_return_vec is None:
                ep_return_vec = np.array(reward_vec, dtype=np.float32)
            else:
                ep_return_vec += reward_vec

            state = next_state
            ep_len += 1

            # Perform learning updates
            if self.trained_timesteps >= start_timesteps and self.trained_timesteps % self.update_every_n_steps == 0:
                for _ in range(self.updates_per_session):
                    self.worker.update()
            
            self.trained_timesteps += 1

            # Save a resume snapshot periodically.
            if self.trained_timesteps > 0 and self.trained_timesteps % self.timesteps_between_checkpoints == 0 and self.checkpointer:
                manager_state = {
                    'trained_timesteps': self.trained_timesteps,
                    'training_stats': self._training_stats
                }
                self.checkpointer.save_island_snapshot(
                    agent=self.worker.agent,
                    buffer=self.worker.buffer(),
                    manager_state=manager_state,
                    island_id=self.island_id
                )

            # Handle episode termination
            if done or trunc:
                scalar_return = (ep_return_vec * self.w).sum()
                print(f"[Island {self.island_id}] Steps: {self.trained_timesteps}, Ep: {episodes_completed+1}, Len: {ep_len}, Return: {scalar_return:.2f}")
                
                self.scalar_returns.append(scalar_return)
                self.vector_returns.append(ep_return_vec)
                
                # Reset for next episode
                state, _ = self.env.reset()
                ep_return_vec = None
                ep_len = 0
                episodes_completed += 1
            
            # Update persistent variables
            self._training_stats.set(state, ep_return_vec, ep_len, episodes_completed, random_action)
        
        return self.trained_timesteps

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
        buffer : IDK #NOTE
            Buffer for this island.
        w : np.ndarray
            The scalarising weight vector for this island.
        """
        self._eval_policy()
        self.pop=[self._make_rl_actor()]
        # eval_pop.eval_pop(self.pop, eval_env, [1,1], episodes_per_actor=10, max_ep_len=self.max_ep_len)
        return self.pop, self.island_id, self.worker.critic(), self.worker.buffer(), self.w
