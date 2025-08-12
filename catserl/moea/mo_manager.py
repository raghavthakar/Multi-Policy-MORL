# catserl/moea/mo_manager.py

from __future__ import annotations

import itertools
from collections import deque
from pathlib import Path
from typing import Generator, Optional, Tuple, List
import numpy as np

import torch

from catserl.orchestrator.checkpoint import Checkpoint
from catserl.shared.evo_utils import eval_pop
from catserl.island.genetic_actor import GeneticActor


__all__ = ["MOManager"]


class MOManager:
    """
    Manages the multi-objective (MO) stage of training.

    This class loads a checkpoint and provides an `evolve` method.
    """
    def __init__(self, env, ckpt_path: str | Path, device: torch.device | str = "cpu") -> None:
        """
        Initializes the manager by loading a merged checkpoint.

        Parameters
        ----------
        env:
            mo_gym environemnt
        ckpt_path : str | Path
            Path to the merged checkpoint file created by Checkpoint.save_merged().
        device : torch.device | str
            The device to load tensors onto.
        """
        self.env = env
        self.device = torch.device(device)
        self.ckpt_path = Path(ckpt_path).resolve()
        print(f"[MOManager] Loading merged checkpoint from: {self.ckpt_path}")

        ckpt = Checkpoint(self.ckpt_path)
        pop, critics_dict, weights_by_island, meta = ckpt.load_merged(
            device=self.device,
            path=self.ckpt_path,
        )

        self.population = pop
        self.critics = critics_dict
        self.weights_by_island = weights_by_island
        self.meta = meta

        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics.")
    
    # ------------------------------------------------------------------ #
    # --- Stage 2: Step 1 Implementation (New Helper Method) ---
    # ------------------------------------------------------------------ #
    def _find_gap_and_select_parents(
        self, population: List[GeneticActor]
    ) -> Tuple[Optional[GeneticActor], Optional[GeneticActor]]:
        """
        Finds the largest gap in the objective space and returns the two
        actors that define that gap.
        """
        # 1. Extract actors with valid return vectors
        evaluated_actors = [p for p in population if p.vector_return is not None]

        if len(evaluated_actors) < 2:
            print("[MOManager] Not enough evaluated actors to find a gap.")
            return None, None

        # 2. Sort actors based on the first objective
        # Creates a list of (actor, return_vector) tuples
        actor_returns = [(p, p.vector_return) for p in evaluated_actors]
        actor_returns.sort(key=lambda x: x[1][0])

        # 3. Find the pair with the maximum Euclidean distance
        max_dist = -1.0
        parent_idx = -1
        for i in range(len(actor_returns) - 1):
            vec_a = actor_returns[i][1]
            vec_b = actor_returns[i+1][1]
            dist = np.linalg.norm(vec_a - vec_b)
            
            if dist > max_dist:
                max_dist = dist
                parent_idx = i
        
        if parent_idx == -1:
            # This should not happen if we have at least 2 actors
            return None, None

        # 4. Select the two actors that form the largest gap
        parent_a = actor_returns[parent_idx][0]
        parent_b = actor_returns[parent_idx + 1][0]
        
        return parent_a, parent_b

    def evolve(self):
        """
        Evolve the population for one generation.
        """
        # --- This call is preserved as requested ---
        # It populates the .vector_return attribute for each actor
        _, _ = eval_pop.eval_pop(self.population, self.env, [1, 1, 1])

        print("\n--- Starting Stage 2: Step 1 ---")
        
        # --- New code for Step 1 starts here ---
        parent_a, parent_b = self._find_gap_and_select_parents(self.population)

        if parent_a and parent_b:
            print(f"Largest gap found: {parent_a.vector_return} <--> {parent_b.vector_return}")
            print(f"Selected Parent A (ID: {parent_a.pop_id})")
            print(f"Selected Parent B (ID: {parent_b.pop_id})")
        else:
            print("Could not select parents.")

        print("--- Step 1 Complete ---\n")