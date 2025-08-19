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
from torch.utils.data import TensorDataset, DataLoader

from catserl.orchestrator.checkpoint import Checkpoint
from catserl.shared.evo_utils.eval_pop import eval_pop
from catserl.shared.actors import Actor  # Updated import
from catserl.shared.buffers import MiniBuffer
from catserl.moea.basic_visualizer import BasicVisualizer


__all__ = ["MOManager"]


class MOManager:
    """Manages the multi-objective (MO) stage of training."""

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

        print(
            f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics."
        )
        print(f"[MOManager] Detected {self.num_objectives} objectives.")

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
            max_steps=parent_a.buffer.max_steps,
        )
        num_samples = mixed_buffer.max_steps // 2

        if len(parent_a.buffer) > 0:
            s, a, r, s2, d = parent_a.buffer.sample(
                min(num_samples, len(parent_a.buffer)), device="cpu"
            )
            mixed_buffer.add_batch(s, a, r, s2, d)

        if len(parent_b.buffer) > 0:
            s, a, r, s2, d = parent_b.buffer.sample(
                min(num_samples, len(parent_b.buffer)), device="cpu"
            )
            mixed_buffer.add_batch(s, a, r, s2, d)

        child.buffer = mixed_buffer
        return child

    def _finetune_child(
        self, child: Actor, target_scalarisation: np.ndarray, config: dict
    ):
        """Fine-tunes the child's policy using its mixed buffer."""
        # --- 1. Setup ---
        optimizer = torch.optim.Adam(child.policy.parameters(), lr=config.get("lr", 3e-4))
        num_epochs = config.get("epochs", 250)
        batch_size = config.get("batch_size", 256)
        beta = config.get("awr_beta", 5.0)

        if len(child.buffer) < batch_size:
            print("[MOManager] Child buffer too small for a full batch. Skipping.")
            return

        critic_ids = sorted(self.critics.keys())
        weights = (
            torch.from_numpy(target_scalarisation).float().to(self.device).unsqueeze(1)
        )

        # --- 2. Pre-compute Advantages for the entire static buffer ---
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=self.device)

        with torch.no_grad():
            per_critic_advs = []
            for cid in critic_ids:
                critic = self.critics[cid]
                v = critic.value(all_s)
                q_sa = critic(all_s).gather(1, all_a.long().unsqueeze(1)).squeeze(1)
                per_critic_advs.append(q_sa - v)

            advs = torch.stack(per_critic_advs, dim=0)
            mins = advs.min(dim=1, keepdim=True).values
            maxs = advs.max(dim=1, keepdim=True).values
            norm_advs = (advs - mins) / (maxs - mins).clamp(min=1e-8)
            hybrid_advantages = torch.sum(norm_advs * weights, dim=0)

        dataset = TensorDataset(all_s, all_a, hybrid_advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- 3. AWR Update Loop ---
        child.policy.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for s_batch, a_batch, adv_batch in dataloader:
                adv_batch = adv_batch.detach()
                log_probs = child.policy.get_log_prob(s_batch, a_batch)

                awr_weights = torch.exp(adv_batch / beta).clamp(max=100.0)
                policy_loss = - (awr_weights * log_probs).mean()

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(child.policy.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += policy_loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch + 1}/{num_epochs}] Avg AWR Loss: {avg_loss:.4f}")
        child.policy.eval()

    def evolve(self, finetune_config: Optional[dict] = None):
        """Evolves the population for one generation."""
        if finetune_config is None:
            finetune_config = {"epochs": 50, "lr": 3e-4, "batch_size": 256}

        print("\n--- Starting MO Stage: Parent Selection ---")
        eval_pop(self.population, self.env, [1 for i in range(self.num_objectives)], episodes_per_actor=50)
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