from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import torch
import numpy as np

from catserl.orchestrator.checkpoint import Checkpoint
from catserl.shared.evo_utils import eval_pop
from catserl.island.genetic_actor import GeneticActor
from catserl.shared.data.buffers import MiniBuffer


__all__ = ["MOManager", "NormalizationStats"]


@dataclass
class NormalizationStats:
    """Holds min/max statistics for normalization."""
    reward_min: np.ndarray
    reward_max: np.ndarray
    value_min: np.ndarray
    value_max: np.ndarray


class MOManager:
    """
    Manages the multi-objective (MO) stage of training.
    """
    def __init__(self, env, num_objectives: int, ckpt_path: str | Path, device: torch.device | str = "cpu") -> None:
        """
        Initializes the manager by loading a merged checkpoint.
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
        
        self.num_objectives = num_objectives


        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics.")
        print(f"[MOManager] Detected {self.num_objectives} objectives.")

    def _find_gap_and_select_parents(
        self, population: List[GeneticActor]
    ) -> Tuple[Optional[GeneticActor], Optional[GeneticActor]]:
        """
        Finds the largest gap in the objective space and returns the two
        actors that define that gap.
        """
        evaluated_actors = [p for p in population if p.vector_return is not None]

        if len(evaluated_actors) < 2:
            print("[MOManager] Not enough evaluated actors to find a gap.")
            return None, None

        actor_returns = [(p, p.vector_return) for p in evaluated_actors]
        actor_returns.sort(key=lambda x: x[1][0])

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
            return None, None

        parent_a = actor_returns[parent_idx][0]
        parent_b = actor_returns[parent_idx + 1][0]
        
        return parent_a, parent_b
    
    def evolve(self):
        """
        Evolve the population for one generation.
        """
        # This call populates the .vector_return attribute for each actor.
        _, _ = eval_pop.eval_pop(self.population, self.env, [1, 1])

        print("\n--- Starting Stage 2: Step 1 ---")
        parent_a, parent_b = self._find_gap_and_select_parents(self.population)

        if not (parent_a and parent_b):
            print("Could not select parents. Aborting evolution step.")
            return

        print(f"Largest gap found: {parent_a.vector_return} <--> {parent_b.vector_return}")
        print(f"Selected Parent A (ID: {parent_a.pop_id})")
        print(f"Selected Parent B (ID: {parent_b.pop_id})")
        print("--- Step 1 Complete ---\n")
