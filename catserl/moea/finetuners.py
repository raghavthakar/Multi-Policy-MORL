# catserl/moea/finetuners.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

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
        if strategy_name == 'dqn':
            return DiscreteAWRFinetuner()
        # --- MODIFIED: Add TD3 strategy ---
        elif strategy_name == 'td3':
            return ContinuousAWRFinetuner()
        elif strategy_name == 'weightedmse':
            return ContinuousWeightedMSEFinetuner()
        else:
            raise ValueError(f"Unknown finetuning strategy for RL algorithm: {strategy_name}")


class DiscreteAWRFinetuner(Finetuner):
    """
    Finetunes a child actor using Advantage-Weighted Regression (AWR).
    This strategy is specifically designed for discrete-action, DQN-style critics.
    """
    def execute(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module],
        config: Dict
    ) -> None:
        """Fine-tunes the child's policy using its mixed buffer."""
        device = next(child.policy.parameters()).device
        # --- 1. Setup ---
        optimizer = torch.optim.Adam(child.policy.parameters(), lr=config.get("lr", 3e-4))
        num_epochs = config.get("epochs", 250)
        batch_size = config.get("batch_size", 256)
        beta = config.get("awr_beta", 5.0)

        if len(child.buffer) < batch_size:
            print("[DiscreteAWRFinetuner] Child buffer too small for a full batch. Skipping.")
            return

        critic_ids = sorted(critics.keys())
        weights = (
            torch.from_numpy(target_scalarisation).float().to(device).unsqueeze(1)
        )

        # --- 2. Pre-compute Advantages for the entire static buffer ---
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=device)

        with torch.no_grad():
            per_critic_advs = []
            for cid in critic_ids:
                critic = critics[cid]
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
            if epoch % 50 == 0 or epoch == num_epochs - 1:
                 print(f"[Epoch {epoch + 1}/{num_epochs}] Avg AWR Loss: {avg_loss:.4f}")
        child.policy.eval()


# --- NEW CLASS FOR TD3 ---
class ContinuousAWRFinetuner(Finetuner):
    """
    Finetunes a continuous-action child actor using Advantage-Weighted Regression (AWR),
    as described in the original AWR paper[cite: 2]. This involves a weighted maximum
    likelihood update that is compatible with TD3-style critics.
    """
    def execute(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module],
        config: Dict
    ) -> None:
        """Fine-tunes the child's continuous policy using its mixed buffer."""
        device = next(child.policy.parameters()).device

        # --- 1. Setup ---
        finetuner_lr = float(config.get("lr", 3e-4))
        optimizer = torch.optim.Adam(child.policy.parameters(), finetuner_lr)
        num_epochs = config.get("epochs", 50)
        batch_size = config.get("batch_size", 256)
        beta = config.get("awr_beta", 1.0) # Temperature parameter
        awr_clip = config.get("awr_clip", 20.0) # Weight clipping from paper [cite: 136]

        if len(child.buffer) < batch_size:
            print("[ContinuousAWRFinetuner] Child buffer too small for a full batch. Skipping.")
            return

        critic_ids = sorted(critics.keys())
        weights = (
            torch.from_numpy(target_scalarisation).float().to(device).unsqueeze(1)
        )

        # --- 2. Pre-compute Hybrid Advantages for the entire static buffer ---
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=device)
        
        with torch.no_grad():
            per_critic_advs = []
            for cid in critic_ids:
                critic = critics[cid]
                # A(s,a) = Q(s,a) - V(s)
                # Estimate V(s) using the critic and the child's current policy action π(s)
                q_values = critic.Q1(all_s, all_a).squeeze(-1)
                policy_actions = child.policy(all_s)
                v_values = critic.Q1(all_s, policy_actions).squeeze(-1)
                advs_i = q_values - v_values
                per_critic_advs.append(advs_i)

            # Stack, normalize per objective, and compute weighted sum for hybrid advantage
            advs = torch.stack(per_critic_advs, dim=0)
            mins = advs.min(dim=1, keepdim=True).values
            maxs = advs.max(dim=1, keepdim=True).values
            norm_advs = (advs - mins) / (maxs - mins).clamp(min=1e-8)
            hybrid_advantages = torch.sum(norm_advs * weights, dim=0)

        dataset = TensorDataset(all_s, all_a, hybrid_advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- 3. AWR Update Loop (Weighted Maximum Likelihood) ---
        child.policy.train()
        print(f"[ContinuousAWRFinetuner] Starting fine-tuning for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            total_loss = 0
            for s_batch, a_batch, adv_batch in dataloader:
                # This call requires adding get_log_prob to your ContinuousPolicy
                log_probs = child.policy.get_log_prob(s_batch, a_batch)

                # Calculate weights as per AWR paper: exp(A(s,a) / beta)
                awr_weights = torch.exp(adv_batch / beta).clamp(max=awr_clip)
                
                # The policy loss is the negative weighted log-likelihood [cite: 53]
                policy_loss = - (awr_weights * log_probs).mean()

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(child.policy.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += policy_loss.item()

            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                 print(f"  [Epoch {epoch + 1}/{num_epochs}] Avg AWR Loss: {avg_loss:.4f}")
        child.policy.eval()

# --- NEW: Deterministic-friendly, Advantage-Weighted MSE Finetuner for TD3 ---
class ContinuousWeightedMSEFinetuner(Finetuner):
    """
    Fine-tunes a continuous (deterministic) policy using an AWR-style
    advantage weighting but a weighted-MSE regression objective instead of
    log-probabilities. This is compatible with TD3-style critics and your
    current deterministic ContinuousPolicy.

    Key details:
      - Uses hybrid advantages built from *per-objective* critics.
      - Uses min over twin Q-heads when available (TD3 consistency).
      - Normalises advantages per-objective (z-score or minmax).
      - Converts A into weights via exp(A/beta) with numerically stable centering.
      - Optimises a weighted MSE: ||mu(s) - a||^2 scaled by those weights.

    Expected critic API:
      - Prefer critic.Q1(s,a), critic.Q2(s,a)
      - Fallback: critic(s,a) -> (q1, q2) or a single q
    """

    def execute(
        self,
        child: Actor,
        target_scalarisation: np.ndarray,
        critics: Dict[int, torch.nn.Module],
        config: Dict
    ) -> None:
        device = next(child.policy.parameters()).device

        # --- 1) Hyperparameters / config ---
        lr          = float(config.get("lr", 3e-4))
        num_epochs  = int(config.get("epochs", 50))
        batch_size  = int(config.get("batch_size", 256))
        beta        = float(config.get("awr_beta", 1.0))          # temperature
        awr_clip    = float(config.get("awr_clip", 20.0))         # weight clipping
        sigma2      = float(config.get("implicit_gaussian_var", 1.0))  # implicit variance for MSE~NLL
        adv_norm    = str(config.get("adv_norm", "zscore")).lower()    # "zscore" | "minmax" | "none"

        if len(child.buffer) < batch_size:
            print("[ContinuousWeightedMSEFinetuner] Child buffer too small for a full batch. Skipping.")
            return

        optimizer = torch.optim.Adam(child.policy.parameters(), lr=lr)

        critic_ids = sorted(critics.keys())
        w = torch.from_numpy(target_scalarisation).float().to(device).unsqueeze(1)

        # --- 2) Fetch full static dataset once ---
        all_s, all_a, _, _, _ = child.buffer.sample(len(child.buffer), device=device)

        # Optional: one-time dataset shuffle to avoid block ordering by source
        perm = torch.randperm(all_s.shape[0], device=device)
        all_s, all_a = all_s[perm], all_a[perm]

        # --- Helpers for TD3 twin-critic min and hybrid advantage ---
        @torch.no_grad()
        def _min_q(critic, s, a):
            """Return min over twin Q-heads."""
            out = critic(s, a)
            q1, q2 = out[0], out[1]
            return torch.min(q1, q2).squeeze(-1)

        @torch.no_grad()
        def _compute_hybrid_advantages():
            per_obj_advs = []
            mu_all = child.policy(all_s)  # deterministic policy output (mu(s))
            for cid in critic_ids:
                critic = critics[cid]
                critic.eval()

                q_sa = _min_q(critic, all_s, all_a)      # Q(s, a)
                v_s  = _min_q(critic, all_s, mu_all)     # V(s) ≈ Q(s, mu(s))
                adv_i = q_sa - v_s                       # A_i(s,a)
                per_obj_advs.append(adv_i)

            advs = torch.stack(per_obj_advs, dim=0)  # [num_obj, N]

            # Per-objective normalisation to align scales
            if adv_norm == "zscore":
                mu  = advs.mean(dim=1, keepdim=True)
                std = advs.std(dim=1, keepdim=True).clamp_min(1e-6)
                norm_advs = (advs - mu) / std
            elif adv_norm == "minmax":
                mins = advs.min(dim=1, keepdim=True).values
                maxs = advs.max(dim=1, keepdim=True).values
                norm_advs = (advs - mins) / (maxs - mins).clamp_min(1e-8)
            else:  # "none"
                norm_advs = advs

            # Hybrid advantage: weighted sum across objectives
            hybrid_adv = torch.sum(norm_advs * w, dim=0)  # [N]
            return hybrid_adv

        with torch.no_grad():
            hybrid_advantages = _compute_hybrid_advantages()

        dataset = TensorDataset(all_s, all_a, hybrid_advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- 3) Weighted-MSE training loop ---
        child.policy.train()
        print(f"[ContinuousWeightedMSEFinetuner] Starting WMSE fine-tuning for {num_epochs} epochs "
              f"(beta={beta}, clip={awr_clip}, norm={adv_norm}, sigma2={sigma2})...")

        for epoch in range(num_epochs):
            total_loss = 0.0
            for s_batch, a_batch, adv_batch in dataloader:
                # Numerically-stable AWR weights: exp( (A / beta) - max )
                x = adv_batch / beta
                x = x - x.max().detach()
                awr_weights = torch.exp(x).clamp(max=awr_clip)  # [B]

                mu = child.policy(s_batch)                      # [B, act_dim]
                sq_err = ((mu - a_batch) ** 2).sum(dim=1)       # [B]
                # Gaussian NLL up to constant factor -> (1 / (2*sigma^2)) * ||a - mu||^2
                loss = (awr_weights * (sq_err / (2.0 * sigma2))).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(child.policy.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / max(1, len(dataloader))
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f"  [Epoch {epoch + 1}/{num_epochs}] Avg WMSE Loss: {avg_loss:.4f}")

        child.policy.eval()
