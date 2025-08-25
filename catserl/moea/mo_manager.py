# mo_manager.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import uuid
import random

from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np

from catserl.orchestrator.checkpoint import Checkpoint
from catserl.shared.evo_utils.eval_pop import eval_pop
from catserl.shared.actors import Actor
from catserl.shared.buffers import MiniBuffer
from catserl.moea.basic_visualizer import BasicVisualizer
from catserl.moea.finetuners import Finetuner


__all__ = ["MOManager"]


class MOManager:
    """Manages the multi-objective (MO) stage of training."""

    # In MOManager class

    def __init__(
        self,
        env,
        num_objectives: int,
        ckpt_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initializes the manager by loading a merged checkpoint."""
        self.env = env
        self.device = torch.device(device)
        self.ckpt_path = Path(ckpt_path).resolve()
        print(f"[MOManager] Loading merged checkpoint from: {self.ckpt_path}")

        ckpt = Checkpoint(self.ckpt_path)
        pop, critics_dict, _, _ = ckpt.load_merged(
            device=self.device,
            path=self.ckpt_path,
        )

        self.population = pop
        self.critics = critics_dict
        self.num_objectives = num_objectives

        # --- MODIFIED: Dynamically determine the algorithm from the loaded population ---
        if not self.population:
            raise ValueError("Cannot initialize MOManager: The loaded population is empty.")
        
        # Inspect the first actor to determine the algorithm type for this session
        self.rl_alg = self.population[0].kind
        # --- END MODIFICATION ---

        # Create finetuner via factory
        self.finetuner = Finetuner.create(self.rl_alg)

        print(
            f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics."
        )
        print(f"[MOManager] Detected {self.num_objectives} objectives.")
        # This will now correctly report 'DQN' or 'TD3'
        print(f"[MOManager] Using finetuning strategy for '{self.rl_alg.upper()}' algorithm.")


        self.generation = 0
        self.visualizer = BasicVisualizer(num_objectives=self.num_objectives)

    def _find_gap_and_select_parents(
        self, population: List[Actor]
    ) -> Tuple[Optional[Actor], Optional[Actor]]:
        """
        Finds the largest gap in the objective space and returns the two
        actors that define that gap.
        """
        actors = [p for p in population if p.vector_return is not None]

        if len(actors) < 2:
            print("[MOManager] Not enough evaluated actors to find a gap.")
            return None, None

        returns_matrix = np.array([p.vector_return for p in actors])
        dist_matrix = squareform(pdist(returns_matrix, "euclidean"))
        np.fill_diagonal(dist_matrix, np.inf)

        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        parent_a_idx = np.argmax(nearest_neighbor_dists)
        parent_b_idx = np.argmin(dist_matrix[parent_a_idx])

        return actors[parent_a_idx], actors[parent_b_idx]

    def _create_offspring(self, parent_a: Actor, parent_b: Actor) -> Actor:
        """
        Creates a child actor with a mixed buffer and a cloned network from a random parent.
        """
        template_parent = random.choice([parent_a, parent_b])
        child = template_parent.clone()
        child.pop_id = uuid.uuid4().hex[:8]

        # Create a new buffer by mixing samples from both parents.
        mixed_buffer = MiniBuffer(
            obs_shape=self.env.observation_space.shape,
            action_type=parent_a.action_type,
            action_dim=parent_a.action_dim,
            max_steps=parent_a.buffer.max_steps,
        )
        num_samples = mixed_buffer.max_steps // 2

        if len(parent_a.buffer) > 0:
            s, a, r, s2, d = parent_a.buffer.sample(
                min(num_samples, len(parent_a.buffer)), device="cpu"
            )
            # The sample method may return numpy arrays or torch tensors depending on implementation
            # add_batch expects numpy arrays, so we ensure they are numpy
            mixed_buffer.add_batch(
                s.cpu().numpy() if isinstance(s, torch.Tensor) else s,
                a.cpu().numpy() if isinstance(a, torch.Tensor) else a,
                r.cpu().numpy() if isinstance(r, torch.Tensor) else r,
                s2.cpu().numpy() if isinstance(s2, torch.Tensor) else s2,
                d.cpu().numpy() if isinstance(d, torch.Tensor) else d
            )


        if len(parent_b.buffer) > 0:
            s, a, r, s2, d = parent_b.buffer.sample(
                min(num_samples, len(parent_b.buffer)), device="cpu"
            )
            mixed_buffer.add_batch(
                s.cpu().numpy() if isinstance(s, torch.Tensor) else s,
                a.cpu().numpy() if isinstance(a, torch.Tensor) else a,
                r.cpu().numpy() if isinstance(r, torch.Tensor) else r,
                s2.cpu().numpy() if isinstance(s2, torch.Tensor) else s2,
                d.cpu().numpy() if isinstance(d, torch.Tensor) else d
            )

        child.buffer = mixed_buffer
        return child

    def _finetune_child(
        self, child: Actor, target_scalarisation: np.ndarray, config: dict
    ):
        """Delegates the finetuning task to the configured strategy object."""
        # MODIFIED: All logic is now external. This method just calls it.
        self.finetuner.execute(child, target_scalarisation, self.critics, config)

    def evolve(self, finetune_config: Optional[dict] = None):
        """Evolves the population for one generation."""
        if finetune_config is None:
            finetune_config = {"epochs": 50, "lr": 3e-3, "batch_size": 256}

        print("\n--- Starting MO Stage: Parent Selection ---")
        eval_pop(self.population, self.env, [1 for i in range(self.num_objectives)], episodes_per_actor=10)
        self.generation += 1
        self.visualizer.update(population=self.population, generation=self.generation)

        parent_a, parent_b = self._find_gap_and_select_parents(self.population)
        if not (parent_a and parent_b):
            print("Could not select parents. Aborting evolution step.")
            return

        print(f"Largest gap: {parent_a.vector_return} <-> {parent_b.vector_return}")
        print(f"Selected Parent A (ID: {parent_a.pop_id}) and B (ID: {parent_b.pop_id})")

        print("\n--- MO Stage: Target Calculation ---")
        evaluated = [p for p in self.population if p.vector_return is not None]
        returns = np.vstack([p.vector_return for p in evaluated])
        mins, maxs = returns.min(axis=0), returns.max(axis=0)
        denom = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
        norm_a = (parent_a.vector_return - mins) / denom
        norm_b = (parent_b.vector_return - mins) / denom
        midpoint = 0.5 * (norm_a + norm_b)
        target_scalarisation = midpoint / np.sum(midpoint) if np.sum(midpoint) > 1e-8 else np.array([0.5] * self.num_objectives)
        print(f"Target weights: {target_scalarisation}")

        print("\n--- MO Stage: Offspring Creation ---")
        child = self._create_offspring(parent_a, parent_b)
        print(f"Created Offspring (ID: {child.pop_id}) with buffer size {len(child.buffer)}.")

        print("\n--- MO Stage: Fine-tuning ---")
        self._finetune_child(child, target_scalarisation, finetune_config)
        self.population.append(child)