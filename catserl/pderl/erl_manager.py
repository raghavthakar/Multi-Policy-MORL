# catserl/erl_manager.py
from __future__ import annotations
from typing import List, Dict
import numpy as np, random, torch
import hashlib

from catserl.shared.envs.four_room import FourRoomWrapper
from catserl.shared.evo_utils import crossover, eval_pop, proximal_mutation, selection
from catserl.shared.rl.dqn import RLWorker
from catserl.shared.envs.rollout import rollout
from catserl.pderl import population


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
        self.cfg = cfg
        self.env = FourRoomWrapper(seed=seed, beta=cfg["env"]["beta_novelty"])
        # -------------------------------------------------------------- #
        # local deterministic RNG for this island
        self.rs = np.random.RandomState(seed)
        # -------------------------------------------------------------- #
        self.w = scalar_weight.astype(np.float32)

        self.worker = RLWorker(self.env.observation_space.shape,
                               self.env.action_space.n,
                               self.w,
                               cfg["dqn"],
                               device)
        
        self.pop = [population.GeneticActor(self.env.observation_space.shape, 
                                            self.env.action_space.n, 
                                            device=device) 
                                            for _ in range(cfg["pderl"]["pop_size"])]
        
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
        rl_actor = population.GeneticActor(self.env.observation_space.shape,
                                           self.env.action_space.n,
                                           hidden_dim=hid,
                                           device=self.worker.device)
        rl_actor.load_flat_params(flat)
        # seed buffer with one rollout so proximal mutation works next gen
        # rollout(self.env, rl_actor, learn=True, max_ep_len=self.max_ep_len)
        return rl_actor

    # ---- helper for greedy parent selection ------------------------- #
    def _pick_parents(self, elite):
        """
        Choose two *distinct* parents from the `elite` list.
        Probability ∝ (mu – rank + 1) where rank=1 is best.
        """
        mu = len(elite)
        ranks = np.arange(1, mu + 1)
        total = mu * (mu + 1) / 2
        probs = (mu - ranks + 1) / total

        idx_a = self.rs.choice(mu, p=probs)
        idx_b = idx_a
        while idx_b == idx_a:
            idx_b = self.rs.choice(mu, p=probs)

        return elite[idx_a], elite[idx_b]

    # ------------------------------------------------------------------ #
    # ----------  Warm-up generation loop  ------------------------------ #
    # ------------------------------------------------------------------ #
    def train_generation(self, dqn_episodes: int = 10, ea_episodes_per_actor: int = 5) -> Dict:
        """
        Collect `dqn_episodes` rollouts for the RL worker, let the RL worker learn online, and 
        perform one generation for the EA by evaluating each actor for ea_episodes_per_actor episodes.

        Returns
        -------
        Dict with simple metrics you can aggregate or log:
            {
              "mean_scalar_return": ... ,
              "episodes": episodes,
              "frames": int,
            }
        """
        # Run the gradient step on the worker
        for _ in range(dqn_episodes):
            ret_vec, ep_len, ext_ret_vec = rollout(
                self.env, self.worker, learn=True, max_ep_len=self.max_ep_len
            )
            ret_scalar = float((ret_vec * self.w).sum())

            self.vector_returns.append(ret_vec)
            self.scalar_returns.append(ret_scalar)
            self.frames_collected += ep_len

        stats, eval_frames = eval_pop.eval_pop(
            self.pop,
            env=self.env,
            weight_vector=self.w,
            episodes_per_actor=ea_episodes_per_actor,
            max_ep_len=self.max_ep_len,
        )
        self.frames_collected += eval_frames

        # ---------- logging ------------------------------------------ #
        for ind in self.pop:
            print(f"Weight {self.w} vector return: {ind.vector_return}, ")

        # ---------- evolutionary step -------------------------------- #
        mu = max(1, len(self.pop) // 3)
        elite = selection.elitist_select(self.pop, mu)

        # decide migration and offspring count
        migrate = (self.gen_counter % self.migrate_every == 0)
        extra_offsprings = 0 if migrate else 1
        target_offsprings = max(0, mu - 1) + extra_offsprings

        # create crossover offspring
        offsprings = []
        while len(offsprings) < target_offsprings and len(elite) > 1:
            pa, pb = self._pick_parents(elite)
            child = crossover.distilled_crossover(
                pa, pb, self.worker.critic(), self.cfg["pderl"], device=self.worker.device
            )
            offsprings.append(child)
        # fallback clones if not enough offspring
        while len(offsprings) < target_offsprings:
            offsprings.append(elite[0].clone())

        # mutate elites
        mutated_elites = [actor.clone() for actor in elite]
        proximal_mutation.proximal_mutate(
            mutated_elites, self.worker.critic(), sigma=self.cfg["pderl"].get("sigma", 0.02)
        )

        # assemble new population
        self.pop = elite + offsprings + mutated_elites

        # append RL actor if migrating
        if migrate:
            rl_actor = self._make_rl_actor()
            self.pop.append(rl_actor)
            # log migration event
            flat_params = None
            if hasattr(rl_actor, "export_flat_params"):
                flat_params = rl_actor.export_flat_params()
            param_hash = (
                hashlib.md5(flat_params.tobytes()).hexdigest()
                if flat_params is not None
                else None
            )
            self.migration_log.append(
                {"generation": self.gen_counter, "param_hash": param_hash}
            )

        self.gen_counter += 1

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

    def get_migration_log(self):
        """Return the migration log for RL→GA events."""
        return self.migration_log
