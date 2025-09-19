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
from catserl.shared.buffers import MiniBuffer
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
        self.cfg = cfg['mopderl']
        self.num_objectives = cfg['env']['num_objectives']
        self.glob_cfg = cfg
        
        # Load all components from the end of the island stage.
        print(f"[MOManager] Loading merged checkpoint from: {ckpt_path}")
        ckpt = Checkpoint(ckpt_path)
        pop, critics, buffers, _, _ = ckpt.load_merged(device=self.device)

        self.population = pop
        self.critics = critics
        self.specialist_buffers = buffers

        if not self.population:
            raise ValueError("Cannot initialize MOManager: loaded population is empty.")

        self.rl_alg = self.population[0].kind
        self.finetuner = Finetuner.create('weightedmse')

        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics, "
              f"and {len(self.specialist_buffers)} specialist buffers.")
        print(f"[MOManager] Using finetuning strategy for '{self.rl_alg.upper()}' algorithm.")

        self.generation = 0
        self.visualizer = BasicVisualizer(num_objectives=self.num_objectives)

    def _get_pareto_front(self, population: List[Actor]) -> List[Actor]:
        """
        Filters a list of actors to return only the Pareto-optimal set.

        An actor is considered Pareto-optimal if no other actor in the population
        is better in at least one objective while being no worse in all others.

        Args:
            population: A list of actors that have been evaluated.

        Returns:
            A list containing only the non-dominated actors from the population.
        """
        pareto_front = []
        for actor_p in population:
            is_dominated = False
            for actor_q in population:
                # An actor cannot be dominated by itself.
                if actor_p is actor_q:
                    continue
                
                # Check if actor_q dominates actor_p. This is true if actor_q's
                # returns are greater than or equal to actor_p's in all objectives,
                # and strictly greater in at least one objective.
                p_returns = actor_p.vector_return
                q_returns = actor_q.vector_return
                if np.all(q_returns >= p_returns) and np.any(q_returns > p_returns):
                    is_dominated = True
                    break  # Found a dominator, no need to check further.
            
            if not is_dominated:
                pareto_front.append(actor_p)
                
        return pareto_front

    def _find_gap_and_select_parents(
        self, population: List[Actor]
    ) -> Tuple[Optional[Actor], Optional[Actor]]:
        """
        Identifies the largest adjacent gap on the Pareto front using the
        'Sort-and-Scan' method for parent selection.

        This method first filters the population to find the non-dominated
        (Pareto-optimal) set of actors. It then sorts this set along one
        objective to establish an ordering and finds the adjacent pair with
        the largest Euclidean distance in the objective space.
        """
        evaluated_actors = [p for p in population if p.vector_return is not None]

        # Step 1: Filter the population to get only the Pareto-optimal actors.
        pareto_actors = self._get_pareto_front(evaluated_actors)
        print(f"[Parent Selection] Found {len(pareto_actors)} Pareto-optimal actors from {len(evaluated_actors)} candidates.")

        if len(pareto_actors) < 2:
            print("[Parent Selection] Not enough Pareto-optimal actors to find a gap.")
            return None, None

        # Step 2: Sort the Pareto front by the first objective to define adjacency.
        # This creates a path along the front, from one extreme to the other.
        pareto_actors.sort(key=lambda p: p.vector_return[0])

        # Step 3: Scan through adjacent pairs in the sorted list to find the
        # pair with the largest Euclidean distance between their returns.
        max_dist = -1.0
        parent_a, parent_b = None, None

        for i in range(len(pareto_actors) - 1):
            p1 = pareto_actors[i]
            p2 = pareto_actors[i+1]
            
            dist = np.linalg.norm(p1.vector_return - p2.vector_return)
            
            if dist > max_dist:
                max_dist = dist
                parent_a = p1
                parent_b = p2
        
        return parent_a, parent_b

    def _create_offspring(
        self,
        parent_a: Actor,
        parent_b: Actor,
        target_scalarisation: np.ndarray,
    ) -> Actor:
        """
        Creates a child actor and a targeted offline dataset for finetuning.

        The child's policy network is initialized by averaging its parents'
        parameters. Its buffer is populated by sampling from the specialist
        replay buffers in proportion to the target scalarisation weights.
        Crucially, it also tracks the origin island of each transition, which
        is required by the finetuner to select the correct in-distribution critic.
        """
        # Initialize child policy by averaging parent parameters.
        child = parent_a.clone()
        child.pop_id = uuid.uuid4().hex[:8]
        
        flatA = parent_a.flat_params()
        flatB = parent_b.flat_params()
        child.load_flat_params(0.5 * (flatA + flatB))

        # Determine sampling ratios from the target weights.
        new_buffer_size = self.cfg['child_buffer_size']
        island_ids = sorted(self.specialist_buffers.keys())
        num_samples_per_buffer = (target_scalarisation * new_buffer_size).astype(int)
        
        print(f"Creating a new static dataset for child (ID: {child.pop_id}) with target size {new_buffer_size}.")

        # Sample transitions and track their origin island.
        all_s, all_a, all_r, all_s2, all_d, all_origins = [], [], [], [], [], []
        for i, island_id in enumerate(island_ids):
            num_to_sample = num_samples_per_buffer[i]
            buffer = self.specialist_buffers[island_id]

            if num_to_sample == 0 or len(buffer) == 0:
                continue
            
            num_to_sample = min(num_to_sample, len(buffer))
            print(f"  - Sampling {num_to_sample} transitions from specialist buffer {island_id}...")

            s, a, r, s2, d = buffer.sample(num_to_sample)
            all_s.append(s.cpu().numpy())
            all_a.append(a.cpu().numpy())
            all_r.append(r.cpu().numpy())
            all_s2.append(s2.cpu().numpy())
            all_d.append(d.cpu().numpy())
            
            # Store the origin island ID for each sampled transition.
            all_origins.append(np.full(num_to_sample, island_id))

        if not all_s:
            print("Warning: No samples were collected for the child's buffer.")
            return child

        # Assemble the final static dataset.
        final_states = np.concatenate(all_s, axis=0)
        final_actions = np.concatenate(all_a, axis=0)
        final_rewards = np.concatenate(all_r, axis=0)
        final_next_states = np.concatenate(all_s2, axis=0)
        final_dones = np.concatenate(all_d, axis=0)
        final_origins = np.concatenate(all_origins, axis=0)

        # Create a new buffer for the child and populate it.
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
        
        # Attach the parallel array of origin IDs to the child. This is required
        # by the finetuner to select the correct critic for each transition.
        child.buffer_origins = final_origins
        
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
        # target_scalarisation = np.array([0.9, 0.1])
        print(f"Calculated target weights: {target_scalarisation}")

        # 3. Offspring Creation
        child = self._create_offspring(parent_a, parent_b, target_scalarisation)
        
        # 4. Fine-tuning
        print("--- Fine-tuning Child Actor ---")
        self.finetuner.execute(
            child=child,
            target_scalarisation=target_scalarisation,
            critics=self.critics,
            config=self.cfg['finetune']
        )
        
        # Add the new, fine-tuned child to the population for the next generation.
        self.population.append(child)