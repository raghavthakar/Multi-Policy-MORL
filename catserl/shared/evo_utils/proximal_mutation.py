# ==================== catserl/pderl/proximal_mutation.py ====================
"""
Proximal-safe mutation (PDERL, Eq. 4)

θ' = θ + (1 / ‖J‖) · ε ,     ε ~ N(0, σ² I)

where  J = ∂Q(s, πθ(s)) / ∂θ   averaged over a mini-batch of states.
"""

from __future__ import annotations
from typing import List
import torch, numpy as np
from catserl.island.population import GeneticActor


# --------------------------------------------------------------------------- #
# helper: ‖J‖_2  (batch-averaged)
# --------------------------------------------------------------------------- #
def _jacobian_norm(actor: GeneticActor,
                   critic: torch.nn.Module,
                   states: torch.Tensor) -> torch.Tensor:
    """
    L2 norm of gradient of the critic value w.r.t actor parameters,
    average over the batch in accordance with the PDERL reference repo.
    """
    logits = actor.net(states)
    probs  = torch.softmax(logits / 0.1, dim=1)      # τ = 0.1
    q_all  = critic(states)                          # [B, |A|]
    q      = (probs * q_all).sum(dim=1).mean()       # scalar

    grads = torch.autograd.grad(q,
                                actor.net.parameters(),
                                retain_graph=False,
                                create_graph=False)

    total = torch.tensor(0.0, device=states.device)
    for g in grads:
        total += (g ** 2).sum()
    return torch.sqrt(total) + 1e-8                      # avoid div-by-0


# --------------------------------------------------------------------------- #
# main API
# --------------------------------------------------------------------------- #
def proximal_mutate(pop: List[GeneticActor],
                    critic: torch.nn.Module,
                    sigma: float = 0.02,
                    batch_size: int = 32,
                    clamp_std: float = 0.1) -> None:
    """
    Mutate each GeneticActor in `pop` in place.

    Parameters
    ----------
    sigma       : std of isotropic noise ε before scaling.
    batch_size  : #states drawn from actor's MiniBuffer to estimate ‖J‖.
    clamp_std   : upper bound on 1/‖J‖ so steps never exceed `clamp_std·σ`.
    """
    for actor in pop:
        if len(actor.buffer) == 0:
            continue                                    # no data yet

        bs = min(batch_size, len(actor.buffer))
        states, _ = actor.buffer.sample(bs, device=actor.device)
        states.requires_grad_(True)

        jac_norm = _jacobian_norm(actor, critic, states)         # scalar
        scale    = torch.clamp(1.0 / jac_norm, max=clamp_std)    # Eq.(4)

        # ε ~ N(0, σ² I)
        noise = torch.randn_like(actor.flat_params()) * sigma
        new_params = actor.flat_params() + scale * noise
        actor.load_flat_params(new_params)
