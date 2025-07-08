# ==================== catserl/pderl/proximal_mutation.py ====================
"""
Proximal‑safe mutation (Eq. 4 in the PDERL paper, AAAI‑20).

The idea: perturb actor parameters by gaussian noise ε, then
scale that noise by 1 / ‖J‖ where

    J = ∂Q(s, π_θ(s)) / ∂θ        (Jacobian w.r.t. actor parameters)
                                  averaged over a batch of states.

Here we implement a *vector‑norm* version: one global scale per actor,
which is what the original PDERL repo uses (see `proximal_mutate()`).
"""

from __future__ import annotations
from typing import List
import torch, numpy as np
from catserl.pderl.population import GeneticActor


def _jacobian_norm(actor: GeneticActor,
                   critic: torch.nn.Module,
                   states: torch.Tensor) -> torch.Tensor:
    """
    Compute ||J||_2 where J = dQ(s, π(s)) / dθ for the given actor.
    We use the squared‐sums of per‑parameter grads, then sqrt.
    """
    # Forward pass: action indices
    logits = actor.net(states)
    act_idx = logits.argmax(dim=1, keepdim=True)           # [B,1]

    # Critic Q-values for chosen actions
    q = critic(states).gather(1, act_idx).mean()           # scalar

    # Gradients w.r.t. actor parameters
    grads = torch.autograd.grad(q,
                                actor.net.parameters(),
                                retain_graph=False,
                                create_graph=False)

    total = torch.tensor(0.0, device=states.device)
    for g in grads:
        total += (g**2).sum()
    return torch.sqrt(total)


def proximal_mutate(pop: List[GeneticActor],
                    critic: torch.nn.Module,
                    sigma: float = 0.02,
                    batch_size: int = 32,
                    clamp_std: float = 0.1):
    """
    Mutate every actor in `pop` *in place* using proximal mutation.

    Parameters
    ----------
    pop : list[GeneticActor]
    critic : Q‑network frozen parameters
    sigma : base Gaussian std before scaling
    batch_size : number of states drawn from the actor's mini‑buffer
    clamp_std : cap on scaling factor to avoid huge updates
    """
    for actor in pop:
        if len(actor.buffer) < batch_size:
            # fallback: random states from replay will be available later
            continue

        states, _ = actor.buffer.sample(batch_size, device=actor.device)
        states.requires_grad_(True)

        jac_norm = _jacobian_norm(actor, critic, states) + 1e-8
        scale = torch.clamp(sigma / jac_norm, max=clamp_std)

        # Gaussian perturbation in flat parameter space
        noise = torch.randn_like(actor.flat_params()) * sigma
        new_params = actor.flat_params() + scale * noise
        actor.load_flat_params(new_params)

