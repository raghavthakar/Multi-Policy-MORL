# mo_manager.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import uuid
import random

from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np
import mo_gymnasium as mo_gym

from catserl.shared.checkpoint import Checkpoint
from catserl.shared.evo_utils.eval_pop import eval_pop
from catserl.shared.actors import Actor
from catserl.shared.buffers import MiniBuffer, ReplayBuffer
from catserl.moea.basic_visualizer import BasicVisualizer
from catserl.moea.finetuners import Finetuner


__all__ = ["MOManager"]


class MOManager:
    """
    Manages the multi-objective (MO) evolutionary stage of the algorithm.

    This class is responsible for loading a population of specialist actors
    from single-objective training, and then iteratively generating new
    "child" actors to fill gaps in the Pareto front.
    """

    def __init__(
        self,
        env,
        cfg: Dict,
        ckpt_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Initializes the manager by loading a merged checkpoint containing
        specialist actors, critics, and their large replay buffers.

        Args:
            env: The multi-objective environment instance.
            cfg: The main configuration dictionary for the run.
            ckpt_path: Path to the checkpoint file from Stage 1.
            device: The torch device to use for computation.
        """
        self.env = env
        self.device = torch.device(device)
        self.cfg = cfg['mopderl']  # Store the relevant config section
        self.num_objectives = cfg['env']['num_objectives']
        self.glob_cfg = cfg
        
        # Load all components from the end of the island stage
        print(f"[MOManager] Loading merged checkpoint from: {ckpt_path}")
        ckpt = Checkpoint(ckpt_path)
        pop, critics, buffers, _, _ = ckpt.load_merged(device=self.device)

        self.population = pop
        self.critics = critics
        self.specialist_buffers = buffers

        if not self.population:
            raise ValueError("Cannot initialize MOManager: loaded population is empty.")

        # Configure the manager based on the loaded actor type (e.g., 'td3')
        self.rl_alg = self.population[0].kind
        self.finetuner = Finetuner.create('weightedmse')

        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics, "
              f"and {len(self.specialist_buffers)} specialist buffers.")
        print(f"[MOManager] Using finetuning strategy for '{self.rl_alg.upper()}' algorithm.")

        self.generation = 0
        self.visualizer = BasicVisualizer(num_objectives=self.num_objectives)

    def _find_gap_and_select_parents(
        self, population: List[Actor]
    ) -> Tuple[Optional[Actor], Optional[Actor]]:
        """
        Identifies the largest gap in the objective space along the Pareto front
        and returns the two actors that define that gap.

        The method calculates pairwise Euclidean distances between all evaluated
        actors and selects the actor with the largest nearest-neighbor distance
        as the first parent. Its nearest neighbor is the second parent.
        """
        evaluated_actors = [p for p in population if p.vector_return is not None]

        if len(evaluated_actors) < 2:
            print("[MOManager] Not enough evaluated actors to find a gap.")
            return None, None

        # Calculate pairwise distances in the objective space
        returns_matrix = np.array([p.vector_return for p in evaluated_actors])
        dist_matrix = squareform(pdist(returns_matrix, "euclidean"))
        np.fill_diagonal(dist_matrix, np.inf) # Exclude self-distance

        # Find the actor with the largest nearest-neighbor distance
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        parent_a_idx = np.argmax(nearest_neighbor_dists)
        parent_b_idx = np.argmin(dist_matrix[parent_a_idx])

        return evaluated_actors[parent_a_idx], evaluated_actors[parent_b_idx]

    def _create_offspring(
        self,
        parent_a: Actor,
        parent_b: Actor,
        target_scalarisation: np.ndarray,
    ) -> Actor:
        """
        Creates a child actor with a targeted offline dataset.

        The child's policy network is cloned from a random parent. Its buffer is
        then populated by sampling from the large specialist replay buffers in
        proportion to the target scalarisation weights. This creates a tailored,
        static dataset for the subsequent fine-tuning step.
        """
        # 1. Initialize child policy by cloning a random parent's network
        template_parent = random.choice([parent_a, parent_a])
        child = template_parent.clone()
        child.pop_id = uuid.uuid4().hex[:8]

        # 2. Determine sampling ratios from the target weights
        new_buffer_size = self.cfg['child_buffer_size']
        island_ids = sorted(self.specialist_buffers.keys())
        num_samples_per_buffer = (target_scalarisation * new_buffer_size).astype(int)
        
        print(f"Creating a new static dataset for child (ID: {child.pop_id}) with target size {new_buffer_size}.")

        # 3. Sample transitions from each specialist buffer
        all_s, all_a, all_r, all_s2, all_d = [], [], [], [], []
        for i, island_id in enumerate(island_ids):
            num_to_sample = num_samples_per_buffer[i]
            buffer = self.specialist_buffers[island_id]

            if num_to_sample == 0 or len(buffer) == 0:
                continue
            
            # Ensure we don't request more samples than are available
            num_to_sample = min(num_to_sample, len(buffer))
            print(f"  - Sampling {num_to_sample} transitions from specialist buffer {island_id}...")

            s, a, r, s2, d = buffer.sample(num_to_sample)
            all_s.append(s.cpu().numpy())
            all_a.append(a.cpu().numpy())
            all_r.append(r.cpu().numpy())
            all_s2.append(s2.cpu().numpy())
            all_d.append(d.cpu().numpy())

        if not all_s:
            print("Warning: No samples were collected for the child's buffer.")
            return child

        # 4. Assemble the final static dataset
        final_states = np.concatenate(all_s, axis=0)
        final_actions = np.concatenate(all_a, axis=0)
        final_rewards = np.concatenate(all_r, axis=0)
        final_next_states = np.concatenate(all_s2, axis=0)
        final_dones = np.concatenate(all_d, axis=0)

        # 5. Create a new buffer for the child and populate it
        child_buffer = MiniBuffer(
            obs_shape=self.env.observation_space.shape,
            action_type=child.action_type,
            action_dim=child.action_dim,
            max_steps=len(final_states),
        )
        child_buffer.add_batch(
            final_states, final_actions, final_rewards, final_next_states, final_dones
        )
        child.buffer = child_buffer
        
        print(f"Child's buffer created with {len(child.buffer)} total transitions.")
        return child

    def evolve(self):
        """
        Executes one full generation of the multi-objective evolutionary algorithm.
        """
        print("\n--- Starting MO Generation ---")
        
        eval_env = mo_gym.make(self.glob_cfg['env']['name'])
        eval_pop([ind for ind in self.population if ind.vector_return is None], eval_env, [0.5,0.5], episodes_per_actor=self.cfg['episodes_per_actor'])

        self.generation += 1
        self.visualizer.update(population=self.population, generation=self.generation)

        # 1. Parent Selection
        parent_a, parent_b = self._find_gap_and_select_parents(self.population)
        if not (parent_a and parent_b):
            print("Could not select parents. Aborting evolution step.")
            return

        print(f"Selected parents with returns: {parent_a.vector_return} <-> {parent_b.vector_return}")

        # 2. Target Calculation
        evaluated = [p for p in self.population if p.vector_return is not None]
        returns = np.vstack([p.vector_return for p in evaluated])
        mins, maxs = returns.min(axis=0), returns.max(axis=0)
        
        denom = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
        norm_a = (parent_a.vector_return - mins) / denom
        norm_b = (parent_b.vector_return - mins) / denom
        
        midpoint = 0.5 * (norm_a + norm_b)
        target_scalarisation = midpoint / np.sum(midpoint) if np.sum(midpoint) > 1e-8 else np.ones(self.num_objectives) / self.num_objectives
        print(f"Calculated target weights: {target_scalarisation}")

        # 3. Offspring Creation
        child = self._create_offspring(parent_a, parent_b, target_scalarisation)
        
        # 4. Fine-tuning
        print("--- Fine-tuning Child Actor ---")
        self.finetuner.execute(
            child=child,
            target_scalarisation=target_scalarisation,
            critics=self.critics,
            config=self.cfg['finetune'] # Pass the finetuning sub-config
        )
        
        # Add the new, fine-tuned child to the population for the next generation
        self.population.append(child)
