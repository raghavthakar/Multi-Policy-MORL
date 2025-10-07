# catserl/pderl/crossover.py
from __future__ import annotations
from typing import Dict
import torch
import numpy as np
import torch.nn.functional as F

from catserl.shared.actors import Actor

def fill_child_buffer_from_parents(child: Actor, parent1: Actor, parent2: Actor):
    """
    Fills a child's buffer with recent transitions from its parents.

    The child's buffer is populated with an equal proportion of the most
    recent experiences from each parent's personal buffer ("genetic memory").
    """
    child_buf_size = child.buffer.max_steps
    num_from_p1 = min(len(parent1.buffer), child_buf_size // 2)
    num_from_p2 = min(len(parent2.buffer), child_buf_size - num_from_p1)

    # Helper to efficiently extract the last N transitions from a MiniBuffer.
    def get_latest_transitions(parent: Actor, n: int):
        buf = parent.buffer
        if n == 0:
            return None
        
        # Determine indices for the last n transitions in the circular buffer.
        end_idx = buf.ptr
        start_idx = end_idx - n
        if start_idx < 0:
            idxs = np.concatenate([np.arange(start_idx + buf.max_steps, buf.max_steps), np.arange(end_idx)])
        else:
            idxs = np.arange(start_idx, end_idx)
        
        return (
            buf.states[idxs], buf.actions[idxs], buf.rewards[idxs],
            buf.next_states[idxs], buf.dones[idxs]
        )

    # Collect transitions and add them to the child's buffer in one batch.
    p1_data = get_latest_transitions(parent1, num_from_p1)
    p2_data = get_latest_transitions(parent2, num_from_p2)
    
    all_data = []
    if p1_data: all_data.append(p1_data)
    if p2_data: all_data.append(p2_data)

    if all_data:
        s, a, r, s2, d = [np.concatenate(item) for item in zip(*all_data)]
        child.buffer.add_batch(s, a, r, s2, d)


def distilled_crossover(
    parent1: Actor,
    parent2: Actor,
    critic: torch.nn.Module,
    main_scalar_weight: np.ndarray,
    cfg: Dict,
    device: torch.device
) -> Actor:
    """
    Performs Q-filtered distillation crossover for continuous (TD3) policies.

    A new child policy is trained to imitate the actions of its parents.
    For each state, the "target" action is the one from the parent that the
    provided critic estimates to have a higher Q-value. The child is trained
    via regression (MSE loss) to mimic this superior behavior.
    """
    bc_epochs = int(cfg.get("crossover_epochs", 12))
    bc_batch_size = int(cfg.get("crossover_batch_size", 128))
    lr = float(cfg.get("crossover_lr", 1e-3))

    # Initialize the child by cloning the fitter parent's policy.
    p1_fit = parent1.fitness if parent1.fitness is not None else -np.inf
    p2_fit = parent2.fitness if parent2.fitness is not None else -np.inf
    fitter_parent = parent1 if p1_fit >= p2_fit else parent2
    child = fitter_parent.clone()
    child.policy.to(device)
    
    # Prepare parent policies and critic for evaluation.
    parent1.policy.to(device).eval()
    parent2.policy.to(device).eval()
    critic.to(device).eval()

    # Create the mixed dataset for the child.
    fill_child_buffer_from_parents(child, parent1, parent2)
    if len(child.buffer) < bc_batch_size:
        return child # Not enough data to train.

    optimizer = torch.optim.Adam(child.policy.parameters(), lr=lr)
    
    # Prepare the scalarization weight for conditioning the critic.
    weight_tensor = torch.from_numpy(main_scalar_weight).float().to(device)

    # Behavior cloning loop to distill parent knowledge.
    for _ in range(bc_epochs):
        states, _ = child.buffer.sample(bc_batch_size, device=device)
        
        with torch.no_grad():
            # Get the continuous actions from both parent policies.
            p1_actions = parent1.policy(states)
            p2_actions = parent2.policy(states)
            
            # Condition the state with the island's main objective weight.
            weight_batch = weight_tensor.expand(states.shape[0], -1)
            state_conditioned = torch.cat([states, weight_batch], 1)

            # Use the critic to determine which parent's action is better.
            q1 = critic.Q1(state_conditioned, p1_actions)
            q2 = critic.Q1(state_conditioned, p2_actions)
            
            # The target is the action from the parent with the higher Q-value.
            target_action = torch.where(q1 >= q2, p1_actions, p2_actions)

        # Train the child to regress onto the target action.
        child_action = child.policy(states)
        loss = F.mse_loss(child_action, target_action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return child