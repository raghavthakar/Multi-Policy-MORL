# catserl/moea/finetuners.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from catserl.shared.actors import Actor


class Finetuner(ABC):
    """
    An abstract base class for all finetuning strategies.
    It defines the contract for how a child actor is improved.
    """
    @abstractmethod
    def offline_finetune(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module]
    ) -> None:
        """
        This method contains the logic for finetuning the child actor.
        It modifies the child actor's policy network in-place.
        """
        pass

    @staticmethod
    def create(strategy_name: str, config: Dict) -> 'Finetuner':
        """
        Factory method to create a finetuner instance from a string name.
        """
        strategy_name = strategy_name.lower()
        if strategy_name == 'weightedmse':
            return ContinuousWeightedMSEFinetuner(config)
        else:
            raise ValueError(f"Unknown finetuning strategy: {strategy_name}")


class ContinuousWeightedMSEFinetuner(Finetuner):
    """
    Fine-tunes a continuous policy using Advantage-Weighted Mean-Squared Error.
    """

    def __init__(self, config: Dict) -> None:
        """
        Initializes the finetuner and sets its hyperparameters from a config dict.
        """
        self.lr = float(config.get("lr", 3e-4))
        self.num_epochs = int(config.get("epochs", 50))
        self.batch_size = int(config.get("batch_size", 256))
        self.beta = float(config.get("awr_beta", 1.0))
        self.awr_clip = float(config.get("awr_clip", 20.0))
        self.adv_norm = str(config.get("adv_norm", "zscore")).lower()

    def offline_finetune(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module]
    ) -> None:
        """
        Performs the core offline finetuning logic on a static dataset.
        
        This method orchestrates the finetuning by preparing the dataset,
        computing advantage estimates, and running the policy update loop.
        """
        device = next(child.policy.parameters()).device
        optimizer = torch.optim.Adam(child.policy.parameters(), lr=self.lr)
        
        if len(child.buffer) < self.batch_size:
            print("[WMSE Finetuner] Child buffer too small for a full batch. Skipping.")
            return

        # Pre-compute all advantage values for the entire static dataset.
        with torch.no_grad():
            hybrid_advantages = self._compute_hybrid_advantages(child, critics, target_scalarisation, device)

        # Run the weighted-MSE training loop to update the child's policy.
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=device)
        dataset = TensorDataset(all_s, all_a, hybrid_advantages)
        
        self._train_loop(child, optimizer, dataset)

    @staticmethod
    @torch.no_grad()
    def _min_q(critic: torch.nn.Module, s_conditioned: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Helper to return the minimum over twin Q-heads from a TD3 critic."""
        q1, q2 = critic(s_conditioned, a)
        return torch.min(q1, q2).squeeze(-1)

    @torch.no_grad()
    def _compute_hybrid_advantages(
        self,
        child: Actor,
        critics: Dict[int, torch.nn.Module],
        target_scalarisation: np.ndarray,
        device: torch.device
    ) -> torch.Tensor:
        """
        Computes advantage estimates for the entire static dataset.
        
        It iterates through each specialist critic and its corresponding in-distribution
        data. For each data subset, it uses the specialist critic to compute the
        advantage for ALL objectives by conditioning its input on the appropriate
        one-hot scalarization vector.
        """
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=device)
        all_origins = torch.from_numpy(child.buffer_origins).long().to(device)
        
        n_samples = all_s.shape[0]
        num_objectives = len(critics)
        one_hot_vectors = torch.eye(num_objectives, device=device)
        
        advs = torch.zeros(num_objectives, n_samples, device=device)
        mu_all = child.policy(all_s)

        for cid, critic in critics.items():
            critic.eval()
            origin_mask = (all_origins == cid)
            if not origin_mask.any():
                continue

            s_subset, a_subset, mu_subset = all_s[origin_mask], all_a[origin_mask], mu_all[origin_mask]
            n_subset = s_subset.shape[0]

            for obj_idx in range(num_objectives):
                w_vec = one_hot_vectors[obj_idx]
                w_batch = w_vec.expand(n_subset, -1)
                s_conditioned = torch.cat([s_subset, w_batch], 1)
                
                q_sa = self._min_q(critic, s_conditioned, a_subset)
                v_s  = self._min_q(critic, s_conditioned, mu_subset)
                advs[obj_idx, origin_mask] = q_sa - v_s

        # Normalize advantages per-objective to align their scales.
        if self.adv_norm == "zscore":
            mu  = advs.mean(dim=1, keepdim=True)
            std = advs.std(dim=1, keepdim=True).clamp_min(1e-6)
            norm_advs = (advs - mu) / std
        elif self.adv_norm == "minmax":
            mins = advs.min(dim=1, keepdim=True).values
            maxs = advs.max(dim=1, keepdim=True).values
            norm_advs = (advs - mins) / (maxs - mins).clamp_min(1e-8)
        else:
            norm_advs = advs
        
        # Create the final hybrid advantage by scalarizing with the target weights.
        w_target = torch.from_numpy(target_scalarisation).float().to(device).unsqueeze(1)
        hybrid_adv = torch.sum(norm_advs * w_target, dim=0)
        return hybrid_adv

    def _train_loop(self, child: Actor, optimizer: torch.optim.Optimizer, dataset: TensorDataset) -> None:
        """
        Runs the weighted-MSE training loop to update the child's policy.
        """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        child.policy.train()
        print(f"[WMSE Finetuner] Starting finetuning for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            total_loss = 0.0
            for s_batch, a_batch, adv_batch in dataloader:
                # Compute advantage-weighted weights using a numerically stable softmax.
                x = adv_batch / self.beta
                x = x - x.max().detach()
                awr_weights = torch.exp(x).clamp(max=self.awr_clip)

                # The loss is the mean-squared error between the policy's action
                # and the buffer action, scaled by the AWR weight.
                mu = child.policy(s_batch)
                mse_loss = ((mu - a_batch) ** 2).sum(dim=1)
                # loss = (awr_weights * mse_loss).mean()
                loss = mse_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(child.policy.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                print(f"  [Epoch {epoch + 1}/{self.num_epochs}] Avg WMSE Loss: {avg_loss:.4f}")

        child.policy.eval()