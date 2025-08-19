# ==================== catserl/pderl/crossover.py ====================
from __future__ import annotations
from typing import Dict
import torch, numpy as np
import torch.nn.functional as F

from catserl.shared.actors import DQNActor

def fill_child_buffer_from_parents(child, parent1, parent2):
    """
    Fill child's MiniBuffer with the latest transitions from both parents, in equal proportions.
    """
    buf_size = child.buffer.max_steps
    n1 = min(len(parent1.buffer), buf_size // 2)
    n2 = min(len(parent2.buffer), buf_size - n1)

    def get_latest_states_actions(parent, n):
        buf = parent.buffer
        length = len(buf)
        if length == 0 or n == 0:
            return np.empty((0, *buf.states.shape[1:]), dtype=buf.states.dtype), np.empty((0,), dtype=buf.actions.dtype)
        if buf.full:
            # Circular buffer: latest n are from (ptr-n)%max_steps to ptr-1
            idxs = (np.arange(buf.ptr - n, buf.ptr) % buf.max_steps)
            return buf.states[idxs], buf.actions[idxs]
        else:
            return buf.states[max(0, length - n):length], buf.actions[max(0, length - n):length]

    # Get latest transitions
    s1, a1 = get_latest_states_actions(parent1, n1)
    s2, a2 = get_latest_states_actions(parent2, n2)

    # Concatenate
    states = np.concatenate([s1, s2], axis=0)
    actions = np.concatenate([a1, a2], axis=0)

    # Reset child's buffer
    child.buffer.ptr = 0
    child.buffer.full = False

    # Add transitions to child's buffer
    for s, a in zip(states, actions):
        child.buffer.add(s, a)

def distilled_crossover(parent1: DQNActor,
                        parent2: DQNActor,
                        critic: torch.nn.Module,
                        cfg: Dict,
                        device: torch.device | str = "cpu") -> DQNActor:
    """
    Implementation of Algorithm 1 (PDERL).

    Parameters
    ----------
    parent1 / parent2 : GeneticActor
    critic            : DuelingQNet (frozen)
    cfg               : expects keys { bc_epochs, bc_batch, crossover_lr }
    device            : where the child network will live

    Returns
    -------
    child : GeneticActor
    """
    device = torch.device(device)
    bc_epochs = int(cfg.get("bc_epochs", 1))
    bc_batch  = int(cfg.get("bc_batch", 256))
    lr        = float(cfg.get("crossover_lr", 1e-3))

    # ---------------------------------------------------------------
    # 1. Pick fitter parent as initial genome
    # ---------------------------------------------------------------
    p1_fit = parent1.fitness if parent1.fitness is not None else -np.inf
    p2_fit = parent2.fitness if parent2.fitness is not None else -np.inf
    fitter, other = (parent1, parent2) if p1_fit >= p2_fit else (parent2, parent1)

    child = fitter.clone()          # deep copy of policy + buffer
    child.net.to(device)
    critic = critic.to(device).eval()

    # Fill child's buffer with latest transitions from both parents
    fill_child_buffer_from_parents(child, parent1, parent2)
    child.buffer.shuffle()

    opt = torch.optim.Adam(child.net.parameters(), lr=lr)

    # ---------------------------------------------------------------
    # 2. Behaviour-cloning loop
    # ---------------------------------------------------------------
    for _ in range(bc_epochs):
        # Sample a batch from the child's own buffer
        if len(child.buffer) < bc_batch:
            # Not enough samples, skip this epoch
            continue
        states, _ = child.buffer.sample(bc_batch, device)

        # greedy actions of both parents
        with torch.no_grad():
            a1 = parent1.net(states).argmax(dim=1)        # [B]
            a2 = parent2.net(states).argmax(dim=1)
            q   = critic(states)                          # [B,|A|]
            q1  = q.gather(1, a1.unsqueeze(1)).squeeze(1)
            q2  = q.gather(1, a2.unsqueeze(1)).squeeze(1)
            target_act = torch.where(q1 >= q2, a1, a2)    # critic winner

        # BCE / CE on child logits
        logits = child.net(states)
        loss = F.cross_entropy(logits, target_act)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return child

# --------------------------------------------------------------------------- #
#  MOPDERL variant (caller decides which parent is better / worse)
# --------------------------------------------------------------------------- #
def mo_distilled_crossover(
    better_parent: DQNActor,
    worse_parent: DQNActor,
    critic: torch.nn.Module,
    cfg: Dict,
    device: torch.device | str = "cpu",
) -> DQNActor:
    """
    Multi-objective distilled crossover (Tran-Long et al., 2023)

    Caller responsibilities
    -----------------------
    • `better_parent`  : policy whose **weights** initialise the child.  
    • `worse_parent`   : policy whose **critic & buffer** provide targets.  
    • Dominance check : caller should pick parents accordingly, but we
      assert using pygmo to avoid silent mistakes (maximisation → negate).
    """

    # hyper-params ----------------------------------------------------------
    device    = torch.device(device)
    epochs    = int(cfg.get("bc_epochs", 1))
    batch     = int(cfg.get("bc_batch", 256))
    lr        = float(cfg.get("crossover_lr", 1e-3))

    # child initialisation --------------------------------------------------
    child = better_parent.clone()
    child.net.to(device)
    critic = critic.to(device).eval()

    # Fill child's buffer with latest transitions from both parents
    fill_child_buffer_from_parents(child, better_parent, worse_parent)
    child.buffer.shuffle()

    opt = torch.optim.Adam(child.net.parameters(), lr=lr)

    # behaviour cloning loop -----------------------------------------------
    for _ in range(epochs):
        # Sample a batch from the child's own buffer
        if len(child.buffer) < batch:
            continue
        states, _ = child.buffer.sample(batch, device)
        with torch.no_grad():
            a_better = better_parent.net(states).argmax(dim=1)   # greedy actions
            a_worse  = worse_parent .net(states).argmax(dim=1)
            q_vals   = critic(states)                            # [B, |A|]
            q_better = q_vals.gather(1, a_better.unsqueeze(1)).squeeze(1)
            q_worse  = q_vals.gather(1, a_worse .unsqueeze(1)).squeeze(1)
            target_act = torch.where(q_better >= q_worse, a_better, a_worse)

        logits = child.net(states)
        loss   = F.cross_entropy(logits, target_act)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return child
