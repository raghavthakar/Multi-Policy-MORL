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
    def execute(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module],
        config: Dict
    ) -> None:
        """
        This method contains the logic for finetuning the child actor.
        It modifies the child actor's policy network in-place.
        """
        pass

    @staticmethod
    def create(strategy_name: str) -> 'Finetuner':
        """
        Factory method to create a finetuner instance from a string name.
        """
        strategy_name = strategy_name.lower()
        if strategy_name == 'weightedmse':
            return ContinuousWeightedMSEFinetuner()
        else:
            raise ValueError(f"Unknown finetuning strategy: {strategy_name}")

class ContinuousWeightedMSEFinetuner(Finetuner):
    """
    Fine-tunes a continuous policy using Advantage-Weighted Mean-Squared Error.

    This method leverages specialist critics that are conditioned on a
    scalarization weight. For each transition in the child's mixed buffer, it
    selects the critic corresponding to the transition's origin to ensure the
    (state, action) pair is in-distribution. It then uses this single critic
    to estimate Q-values for all objectives by providing the appropriate
    one-hot scalarization vectors. This solves the out-of-distribution action
    evaluation problem, leading to more reliable advantage estimates.
    """

    def execute(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module],
        config: Dict
    ) -> None:
        device = next(child.policy.parameters()).device

        # Hyperparameters
        lr          = float(config.get("lr", 3e-4))
        num_epochs  = int(config.get("epochs", 50))
        batch_size  = int(config.get("batch_size", 256))
        beta        = float(config.get("awr_beta", 1.0))
        awr_clip    = float(config.get("awr_clip", 20.0))
        adv_norm    = str(config.get("adv_norm", "none")).lower()

        if len(child.buffer) < batch_size:
            print("[WMSE Finetuner] Child buffer too small for a full batch. Skipping.")
            return

        optimizer = torch.optim.Adam(child.policy.parameters(), lr=lr)
        
        # Prepare the dataset and one-hot vectors for objectives.
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=device)
        all_origins = torch.from_numpy(child.buffer_origins).long().to(device)
        
        num_objectives = len(critics)
        one_hot_vectors = torch.eye(num_objectives, device=device)

        @torch.no_grad()
        def _min_q(critic, s_conditioned, a):
            """Helper to return the minimum over twin Q-heads from a TD3 critic."""
            q1, q2 = critic(s_conditioned, a)
            return torch.min(q1, q2).squeeze(-1)

        @torch.no_grad()
        def _compute_hybrid_advantages():
            """
            Computes advantage estimates for the entire static dataset.
            
            This is the core of the method. It iterates through each specialist
            critic and the data that originated from that critic's island. For
            each of these in-distribution (s,a) pairs, it uses the specialist
            critic to compute the advantage for ALL objectives by conditioning
            its input on the appropriate one-hot scalarization vector.
            """
            n_samples = all_s.shape[0]
            advs = torch.zeros(num_objectives, n_samples, device=device)
            mu_all = child.policy(all_s)

            for cid, critic in critics.items():
                critic.eval()
                # Create a mask to select only the data from this critic's island.
                origin_mask = (all_origins == cid)
                if not origin_mask.any():
                    continue

                # Select the in-distribution subset of data.
                s_subset = all_s[origin_mask]
                a_subset = all_a[origin_mask]
                mu_subset = mu_all[origin_mask]
                n_subset = s_subset.shape[0]

                # For this subset of data, use this critic to calculate the
                # advantage for every objective.
                for obj_idx in range(num_objectives):
                    w_vec = one_hot_vectors[obj_idx]
                    w_batch = w_vec.expand(n_subset, -1)
                    
                    # Condition the state on the one-hot weight vector.
                    s_conditioned = torch.cat([s_subset, w_batch], 1)
                    
                    # A(s,a) = Q(s,a) - V(s), where V(s) is approximated by Q(s, mu(s)).
                    q_sa = _min_q(critic, s_conditioned, a_subset)
                    v_s  = _min_q(critic, s_conditioned, mu_subset)
                    advs[obj_idx, origin_mask] = q_sa - v_s

            # Normalize advantages per-objective to align their scales.
            if adv_norm == "zscore":
                mu  = advs.mean(dim=1, keepdim=True)
                std = advs.std(dim=1, keepdim=True).clamp_min(1e-6)
                norm_advs = (advs - mu) / std
            elif adv_norm == "minmax":
                mins = advs.min(dim=1, keepdim=True).values
                maxs = advs.max(dim=1, keepdim=True).values
                norm_advs = (advs - mins) / (maxs - mins).clamp_min(1e-8)
            else: # "none"
                norm_advs = advs
            
            # Create the final hybrid advantage by scalarizing with the target weights.
            w_target = torch.from_numpy(target_scalarisation).float().to(device).unsqueeze(1)
            hybrid_adv = torch.sum(norm_advs * w_target, dim=0)
            return hybrid_adv

        # Pre-compute all advantage values for the static dataset.
        with torch.no_grad():
            hybrid_advantages = _compute_hybrid_advantages()

        dataset = TensorDataset(all_s, all_a, hybrid_advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- Weighted-MSE Training Loop ---
        child.policy.train()
        print(f"[WMSE Finetuner] Starting finetuning for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            for s_batch, a_batch, adv_batch in dataloader:
                # Compute advantage-weighted weights, using a numerically stable softmax.
                x = adv_batch / beta
                x = x - x.max().detach()
                awr_weights = torch.exp(x).clamp(max=awr_clip)

                # The loss is the mean-squared error between the policy's action
                # and the buffer action, scaled by the AWR weight.
                mu = child.policy(s_batch)
                mse_loss = ((mu - a_batch) ** 2).sum(dim=1)
                loss = (awr_weights * mse_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(child.policy.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  [Epoch {epoch + 1}/{num_epochs}] Avg WMSE Loss: {avg_loss:.4f}")

        child.policy.eval()