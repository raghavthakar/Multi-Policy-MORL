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
    scalar_weight: torch.Tensor
) -> torch.Tensor:
    """
    Computes the L2 norm of the gradient of the critic's Q-value with
    respect to the actor's parameters. This measures the sensitivity
    of the Q-function to changes in the policy.
    """
    # Get the deterministic action from the actor for the given states.
    actions = actor.policy(states)
    
    # Condition the state with the actor's primary objective weight.
    weight_batch = scalar_weight.expand(states.shape[0], -1)
    state_conditioned = torch.cat([states, weight_batch], 1)
    
    # Get the Q-value from the critic for the actor's current actions.
    q_value = critic.Q1(state_conditioned, actions)
    
    # Average the Q-values to get a single scalar for backpropagation.
    q_scalar = q_value.mean()

    # Compute the gradient of this scalar Q-value w.r.t. the actor's parameters.
    grads = torch.autograd.grad(
        q_scalar,
        actor.policy.parameters(),
        retain_graph=False,
        create_graph=False
    )

    # Compute the total L2 norm of the gradient across all parameter tensors.
    total_norm = torch.tensor(0.0, device=states.device)
    for g in grads:
        total_norm += (g ** 2).sum()
        
    return torch.sqrt(total_norm) + 1e-8 # Add epsilon to avoid division by zero.


def proximal_mutate(
    pop: List[Actor],
    critic: torch.nn.Module,
    main_scalar_weight: np.ndarray,
    sigma: float = 0.1,
    batch_size: int = 256
) -> None:
    """
    Mutates each actor in the population in-place using proximal mutation.

    This method adds Gaussian noise to the policy parameters, scaled by the
    inverse of the Q-function's sensitivity to parameter changes. This ensures
    the mutation causes a controlled, non-destructive change in behavior.
    """
    device = critic.l1.weight.device # A way to get the device
    weight_tensor = torch.from_numpy(main_scalar_weight).float().to(device)

    for actor in pop:
        if len(actor.buffer) < batch_size:
            continue # Not enough data in the actor's personal buffer.

        # Sample states from the actor's own recent experiences.
        states, _ = actor.buffer.sample(batch_size, device=device)

        # Calculate the gradient norm (sensitivity).
        grad_norm = _get_q_value_gradient_norm(actor, critic, states, weight_tensor)
        
        # The mutation scale is inversely proportional to the sensitivity.
        scale = 1.0 / grad_norm

        with torch.no_grad():
            # Add scaled Gaussian noise to the actor's flattened parameters.
            flat_params = actor.flat_params()
            noise = torch.randn_like(flat_params) * sigma
            new_params = flat_params + scale * noise
            actor.load_flat_params(new_params)