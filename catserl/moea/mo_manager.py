from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import copy

from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np
import random

from catserl.orchestrator.checkpoint import Checkpoint
from catserl.shared.evo_utils import eval_pop
from catserl.island.genetic_actor import GeneticActor
from catserl.shared.data.buffers import MiniBuffer


__all__ = ["MOManager"]

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
        self, population: List['GeneticActor']
    ) -> Tuple[Optional['GeneticActor'], Optional['GeneticActor']]:
        """
        Finds the largest gap in the multi-objective space and returns the two
        actors that define that gap using the "Largest Nearest-Neighbor Distance" method.
        """
        # Filter for actors that have an assigned performance vector.
        actors = [p for p in population if p.vector_return is not None]

        if len(actors) < 2:
            print("[MOManager] Not enough evaluated actors to find a gap.")
            return None, None

        # Create a NumPy array of the return vectors for efficient calculation.
        returns_matrix = np.array([p.vector_return for p in actors])

        # 1. Calculate the N x N matrix of Euclidean distances between all pairs of points.
        dist_matrix = squareform(pdist(returns_matrix, 'euclidean'))

        # 2. For each point, find its nearest neighbor. To do this, we first set the
        # diagonal to infinity so that a point is not considered its own neighbor.
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find the minimum distance in each row (the distance to the nearest neighbor).
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)

        # 3. Identify which point has the largest nearest-neighbor distance.
        # This point is one of the parents defining the largest local gap.
        parent_a_idx = np.argmax(nearest_neighbor_dists)

        # 4. The other parent is the nearest neighbor to the first parent.
        parent_b_idx = np.argmin(dist_matrix[parent_a_idx])

        # Retrieve the actual actor objects.
        parent_a = actors[parent_a_idx]
        parent_b = actors[parent_b_idx]

        return parent_a, parent_b
    
    def _create_offspring(self, parent_a: GeneticActor, parent_b: GeneticActor):
        # Create the mixed replay buffer for the child's "experience".
        child_buffer = MiniBuffer(obs_shape=self.env.observation_space.shape, max_steps=parent_a.buffer.max_steps)
        # Determine how many samples to take from each parent.
        num_samples_per_parent = child_buffer.max_steps // 2
        # Sample from Parent A's buffer.
        s_a, a_a, r_a, s2_a, d_a = parent_a.buffer.sample(
            min(num_samples_per_parent, len(parent_a.buffer)),
            device=self.device
        )
        for i in range(len(s_a)):
            child_buffer.add(
                s_a[i].cpu().numpy(),
                a_a[i].cpu().numpy(),
                r_a[i].cpu().numpy(),
                s2_a[i].cpu().numpy(),
                d_a[i].cpu().numpy()
            )
        # Sample from Parent B's buffer.
        s_b, a_b, r_b, s2_b, d_b = parent_b.buffer.sample(
            min(num_samples_per_parent, len(parent_b.buffer)),
            device=self.device
        )
        for i in range(len(s_b)):
            child_buffer.add(
                s_b[i].cpu().numpy(),
                a_b[i].cpu().numpy(),
                r_b[i].cpu().numpy(),
                s2_b[i].cpu().numpy(),
                d_b[i].cpu().numpy()
            )

        # Randomly select a parent to act as the template for the network.
        template_parent = random.choice([parent_a, parent_b])
        child = template_parent.clone()
        child.buffer = child_buffer

        return child
    
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

        # --- Step 2: Set target scalarisation ---
        evaluated = [p for p in self.population if p.vector_return is not None]
        if len(evaluated) >= 2:
            returns = np.vstack([p.vector_return for p in evaluated])
            mins = returns.min(axis=0)
            maxs = returns.max(axis=0)

            # Avoid division-by-zero for objectives with no spread
            denom = maxs - mins
            denom = np.where(denom == 0.0, 1.0, denom)

            # Normalise the selected parents (non-mutating)
            norm_a = (parent_a.vector_return - mins) / denom
            norm_b = (parent_b.vector_return - mins) / denom

            # Mid-point (bisection) to be used as target scalarisation
            target_scalarisation = 0.5 * (norm_a + norm_b)

            print(f"Normalised Parent A: {norm_a}")
            print(f"Normalised Parent B: {norm_b}")
            print(f"Target scalarisation (midpoint): {target_scalarisation}")
        else:
            print("[MOManager] Not enough evaluated actors to normalise returns for bisection.")
        
        # --- Step 3: Create a child policy starting point ---
        print("--- Sart step 3 ---")
        child = self._create_offspring(parent_a, parent_b)
        print("Created child: ", child)

        # --- Step 4: Evaluate critics on child's states and normalise ---
        print("--- Start step 4 ---")
        # Sample a batch of transitions from the child's buffer
        if len(child.buffer) == 0:
            print("[MOManager] Child buffer empty; skipping step 4.")
            return

        batch_size = min(1024, len(child.buffer))
        s, _, _, _, _ = child.buffer.sample(batch_size, device=self.device)

        # Compute critic values V(s) per critic and gather stats
        critic_ids = sorted(self.critics.keys())
        if len(critic_ids) == 0:
            print("[MOManager] No critics available; skipping step 4.")
            return

        per_critic_vals = []  # list of tensors [B]
        with torch.no_grad():
            for cid in critic_ids:
                critic = self.critics[cid]
                critic.eval()
                # Prefer a generic value() if available; fall back to Q->max_a
                if hasattr(critic, "value") and callable(getattr(critic, "value")):
                    v = critic.value(s)
                    if v.dim() > 1:
                        v = v.squeeze(-1)  # [B]
                else:
                    q = critic(s)
                    v = q.max(dim=1).values if q.dim() > 1 else q
                per_critic_vals.append(v)

        values = torch.stack(per_critic_vals, dim=0)  # [C, B]

        # Per-critic min and max across the batch
        mins = values.min(dim=1).values  # [C]
        maxs = values.max(dim=1).values  # [C]

        # Remember min and max values per critic
        self._critic_value_mins = {cid: float(mins[i].item()) for i, cid in enumerate(critic_ids)}
        self._critic_value_maxs = {cid: float(maxs[i].item()) for i, cid in enumerate(critic_ids)}

        # Normalise the critic values using per-critic min/max
        denom = (maxs - mins).clamp(min=1e-8)  # [C]
        norm_values = (values - mins.unsqueeze(1)) / denom.unsqueeze(1)  # [C, B]

        # Optional: store normalized values for downstream use
        self._critic_batch_values_norm = {cid: norm_values[i].detach().cpu() for i, cid in enumerate(critic_ids)}

        print("Per-critic value mins:", self._critic_value_mins)
        print("Per-critic value maxs:", self._critic_value_maxs)
        print("Normalised values shape:", tuple(norm_values.shape))
        print("--- Step 4 complete ---")
