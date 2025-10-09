# catserl/pderl/proximal_mutation.py
from __future__ import annotations
from typing import List
import torch
import numpy as np

from catserl.shared.actors import Actor


def _get_q_value_gradient_norm(
    actor: Actor,
    critic: torch.nn.Module,
    states: torch.Tensor,
    scalar_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the L2 norm of the gradient of the critic's Q-value with
    respect to the actor's parameters. This provides a sensitivity measure
    used to scale parameter noise for proximal mutation.
    """
    # Policy inference builds the computation graph from parameters to actions.
    actions = actor.policy(states)

    # Condition the critic on the scalarisation weight and query Q-values.
    weight_batch = scalar_weight.expand(states.shape[0], -1)
    state_conditioned = torch.cat([states, weight_batch], 1)
    q_value = critic.Q1(state_conditioned, actions)

    # Reduce to a scalar objective for gradient backpropagation.
    q_scalar = q_value.mean()

    # Collect only parameters that participate in gradient computation.
    params = [p for p in actor.policy.parameters() if p.requires_grad]

    # Obtain gradients dQ/dθ for all parameters; allow unused in case some
    # parameters do not influence the current forward path.
    grads = torch.autograd.grad(
        q_scalar,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    # Aggregate gradient energy across all tensors.
    total_norm = torch.tensor(0.0, device=states.device)
    for g in grads:
        if g is None:
            continue
        total_norm += (g ** 2).sum()

    return torch.sqrt(total_norm) + 1e-8


def proximal_mutate(
    pop: List[Actor],
    critic: torch.nn.Module,
    main_scalar_weight: np.ndarray,
    sigma: float = 0.1,
    batch_size: int = 256,
) -> None:
    """
    Applies sensitivity-scaled Gaussian perturbations to each actor's
    parameters. The scale is inversely proportional to the critic's
    gradient norm, keeping mutations close in policy space.
    """
    device = next(critic.parameters()).device
    weight_tensor = torch.from_numpy(main_scalar_weight).float().to(device)

    for actor in pop:
        if len(actor.buffer) < batch_size:
            continue

        # Sample states from the actor's personal buffer for sensitivity.
        states, *_ = actor.buffer.sample(batch_size, device=device)

        # Compute scaling via the Q-gradient norm.
        grad_norm = _get_q_value_gradient_norm(actor, critic, states, weight_tensor)
        scale = 1.0 / grad_norm

        # Apply scaled Gaussian noise to the flattened parameter vector.
        with torch.no_grad():
            flat_params = actor.flat_params()
            noise = torch.randn_like(flat_params) * sigma
            actor.load_flat_params(flat_params + scale * noise)
