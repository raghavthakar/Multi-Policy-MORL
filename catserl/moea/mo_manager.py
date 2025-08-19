from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import copy
import uuid

from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader

from catserl.orchestrator.checkpoint import Checkpoint
from catserl.shared.evo_utils import eval_pop
from catserl.shared.actors import DQNActor
from catserl.shared.buffers import MiniBuffer
from catserl.moea.basic_visualizer import BasicVisualizer


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
        pop, critics_dict, _, _ = ckpt.load_merged(
            device=self.device,
            path=self.ckpt_path,
        )

        self.population = pop
        self.critics = critics_dict
        self.num_objectives = num_objectives

        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics.")
        print(f"[MOManager] Detected {self.num_objectives} objectives.")

        self.generation = 0
        self.visualizer = BasicVisualizer(num_objectives=self.num_objectives)

    def _find_gap_and_select_parents(
        self, population: List['DQNActor']
    ) -> Tuple[Optional['DQNActor'], Optional['DQNActor']]:
        """
        Finds the largest gap in the multi-objective space and returns the two
        actors that define that gap using the "Largest Nearest-Neighbor Distance" method.
        """
        actors = [p for p in population if p.vector_return is not None]

        if len(actors) < 2:
            print("[MOManager] Not enough evaluated actors to find a gap.")
            return None, None

        returns_matrix = np.array([p.vector_return for p in actors])
        dist_matrix = squareform(pdist(returns_matrix, 'euclidean'))
        np.fill_diagonal(dist_matrix, np.inf)
        
        nearest_neighbor_dists = np.min(dist_matrix, axis=1)
        parent_a_idx = np.argmax(nearest_neighbor_dists)
        parent_b_idx = np.argmin(dist_matrix[parent_a_idx])

        parent_a = actors[parent_a_idx]
        parent_b = actors[parent_b_idx]

        return parent_a, parent_b
    
    def _create_offspring(self, parent_a: DQNActor, parent_b: DQNActor) -> DQNActor:
        """
        Creates a child actor with a mixed buffer and a cloned network from a random parent.
        """
        # Create the child actor by cloning a random parent
        template_parent = random.choice([parent_a, parent_b])
        child = template_parent.clone()
        child.pop_id = uuid.uuid4().hex[:8] # Assign a new, unique ID

        # Create and populate the mixed replay buffer efficiently
        child_buffer = MiniBuffer(obs_shape=self.env.observation_space.shape, max_steps=parent_a.impl.buffer.max_steps)
        num_samples_per_parent = child_buffer.max_steps // 2

        # Assuming MiniBuffer has an `add_batch` method that takes numpy arrays
        if len(parent_a.impl.buffer) > 0:
            s, a, r, s2, d = parent_a.impl.buffer.sample(min(num_samples_per_parent, len(parent_a.impl.buffer)), device="cpu")
            child_buffer.add_batch(s, a, r, s2, d)
        
        if len(parent_b.impl.buffer) > 0:
            s, a, r, s2, d = parent_b.impl.buffer.sample(min(num_samples_per_parent, len(parent_b.impl.buffer)), device="cpu")
            child_buffer.add_batch(s, a, r, s2, d)
            
        child.impl.buffer = child_buffer
        return child

    def _finetune_child(self, child: DQNActor, target_scalarisation: np.ndarray, config: dict):
        """
        Fine-tunes the child policy using a PPO-style update on the hybrid advantage signal.
        """
        # --- 1. Setup ---
        optimizer = torch.optim.Adam(child.impl.net.parameters(), lr=config.get("lr", 3e-4))
        num_epochs = config.get("epochs", 250)
        batch_size = config.get("batch_size", 256)
        ppo_clip_epsilon = config.get("ppo_clip_epsilon", 0.2)
        
        if len(child.impl.buffer) < batch_size:
            print("[MOManager] Child buffer too small for a full batch. Skipping fine-tuning.")
            return

        critic_ids = sorted(self.critics.keys())
        weights = torch.from_numpy(target_scalarisation).float().to(self.device).unsqueeze(1)
        
        # --- 2. Pre-compute Advantages for the entire static buffer ---
        # This is more stable than re-calculating normalization stats for every batch.
        print("Pre-computing advantages for the entire buffer...")
        all_s, all_a, _, _, _ = child.impl.buffer.sample(len(child.impl.buffer), device=self.device)

        with torch.no_grad():
            per_critic_advs = []
            for cid in critic_ids:
                critic = self.critics[cid]
                critic.eval()
                v = critic.value(all_s)
                q_sa = critic(all_s).gather(1, all_a.long().unsqueeze(1)).squeeze(1)
                per_critic_advs.append(q_sa - v)
            
            advs = torch.stack(per_critic_advs, dim=0) # [C, BufferSize]
            
            # Normalize advantages over the whole buffer
            mins = advs.min(dim=1, keepdim=True).values
            maxs = advs.max(dim=1, keepdim=True).values
            denom = (maxs - mins).clamp(min=1e-8)
            norm_advs = (advs - mins) / denom
            
            # Pre-compute final hybrid advantage and old log probabilities
            hybrid_advantages = torch.sum(norm_advs * weights, dim=0)
            old_log_probs = child.impl.net.get_log_prob(all_s, all_a)

        dataset = TensorDataset(all_s, all_a, old_log_probs, hybrid_advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        
        # --- 3. AWR Update Loop ---
        # NEW: Add the AWR beta hyperparameter
        beta = config.get("awr_beta", 5.0) # A good starting point to tune

        child.impl.net.train()
        for epoch in range(num_epochs): # With AWR, you CAN train for multiple epochs
            total_loss = 0
            for s_batch, a_batch, _, adv_batch in dataloader: # No longer need old_log_prob
                
                # Detach advantages to prevent gradients from flowing into the critics
                adv_batch = adv_batch.detach()

                # Get log-probs from the policy *as it is being updated*
                new_log_prob_batch = child.impl.net.get_log_prob(s_batch, a_batch)
                
                # --- START: AWR Loss Calculation (Replaces PPO Loss) ---
                
                # Calculate the exponential advantage weights
                awr_weights = torch.exp(adv_batch / beta)
                
                # To prevent explosive values, you can clip the weights
                awr_weights = torch.clamp(awr_weights, max=100.0)

                # Calculate the final AWR loss
                policy_loss = - (awr_weights * new_log_prob_batch).mean()
                
                # --- END: AWR Loss Calculation ---

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(child.impl.net.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += policy_loss.item()
            
            print(f"[Epoch {epoch + 1}/{num_epochs}] Average AWR Policy Loss: {total_loss / len(dataloader):.4f}")
        child.impl.net.eval()

    def evolve(self, finetune_config: Optional[dict] = None):
        """
        Evolve the population for one generation.
        """
        if finetune_config is None:
            finetune_config = {"epochs": 50, "lr": 3e-4, "batch_size": 256, "ppo_clip_epsilon": 0.2}

        # --- Step 1: Parent Selection ---
        print("\n--- Starting Stage 2: Step 1 (Parent Selection) ---")
        _, _ = eval_pop.eval_pop(self.population, self.env, [1 for i in range(self.num_objectives)], episodes_per_actor=50)

        # --- ADD YOUR ONE-LINE VISUALIZATION CALL HERE ---
        self.generation += 1
        self.visualizer.update(population=self.population, generation=self.generation)
        # --------------------------------------------------
        
        for ind in self.population:
            print(ind.vector_return)
        parent_a, parent_b = self._find_gap_and_select_parents(self.population)

        if not (parent_a and parent_b):
            print("Could not select parents. Aborting evolution step.")
            return

        print(f"Largest gap found: {parent_a.vector_return} <--> {parent_b.vector_return}")
        print(f"Selected Parent A (ID: {parent_a.pop_id})")
        print(f"Selected Parent B (ID: {parent_b.pop_id})")
        print("--- Step 1 Complete ---\n")

        # --- Step 2: Set Target Scalarisation ---
        print("--- Starting Stage 2: Step 2 (Target Calculation) ---")
        evaluated = [p for p in self.population if p.vector_return is not None]
        target_scalarisation = np.array([0.5] * self.num_objectives)
        if len(evaluated) >= 2:
            returns = np.vstack([p.vector_return for p in evaluated])
            mins, maxs = returns.min(axis=0), returns.max(axis=0)
            denom = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
            norm_a = (parent_a.vector_return - mins) / denom
            norm_b = (parent_b.vector_return - mins) / denom
            midpoint = 0.5 * (norm_a + norm_b)
            if np.sum(midpoint) > 1e-8:
                target_scalarisation = midpoint / np.sum(midpoint) # Project to simplex
            print(f"Target scalarisation (weights): {target_scalarisation}")
        else:
            print("[MOManager] Not enough actors to calculate target; using default.")
        print("--- Step 2 Complete ---\n")

        # --- Step 3: Create Child ---
        print("--- Starting Stage 2: Step 3 (Offspring Creation) ---")
        child = self._create_offspring(parent_a, parent_b)
        print(f"Created Offspring (ID: {child.pop_id}) with buffer size {len(child.impl.buffer)}.")
        print("--- Step 3 Complete ---\n")
        
        # --- Step 4: Fine-tune Child ---
        print("--- Starting Stage 2: Step 4 (Fine-tuning) ---")
        self._finetune_child(child, target_scalarisation, finetune_config)
        print("--- Step 4 Complete ---")

        self.population.append(child)