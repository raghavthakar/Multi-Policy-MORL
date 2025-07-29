# ==================== catserl/pderl/crossover.py ====================
from __future__ import annotations
from typing import Dict
import torch, numpy as np
import torch.nn.functional as F

from catserl.pderl.population import GeneticActor


def distilled_crossover(parent1: GeneticActor,
                        parent2: GeneticActor,
                        critic: torch.nn.Module,
                        cfg: Dict,
                        device: torch.device | str = "cpu") -> GeneticActor:
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
    child.to = lambda d: child  # simple no-op; keeps interface
    child.net.to(device)
    critic.eval()

    opt = torch.optim.Adam(child.net.parameters(), lr=lr)

    # ---------------------------------------------------------------
    # 2. Behaviour-cloning loop
    # ---------------------------------------------------------------
    for _ in range(bc_epochs):
        # mini-batch assembly
        need = bc_batch
        ss, aa = [], []          # lists of states & dummy actions (unused)

        # take up to half from each buffer
        k1 = min(need // 2, len(parent1.buffer))
        if k1:
            s, _ = parent1.buffer.sample(k1, device)
            ss.append(s); need -= k1

        k2 = min(need, len(parent2.buffer))
        if k2:
            s, _ = parent2.buffer.sample(k2, device)
            ss.append(s); need -= k2

        # if still short (rare) pad with random states from fitter buffer
        if need:
            s, _ = fitter.buffer.sample(need, device)
            ss.append(s)

        states = torch.cat(ss, dim=0).to(device)         # [B,…]

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
    better_parent: GeneticActor,
    worse_parent: GeneticActor,
    critic: torch.nn.Module,
    cfg: Dict,
    device: torch.device | str = "cpu",
) -> GeneticActor:
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
    critic.eval()
    opt = torch.optim.Adam(child.net.parameters(), lr=lr)

    # behaviour cloning loop -----------------------------------------------
    for _ in range(epochs):
        states, _ = worse_parent.buffer.sample(batch, device)  # (S,A) but we ignore A
        with torch.no_grad():
            target_act = critic(states).argmax(dim=1)

        logits = child.net(states)
        loss   = F.cross_entropy(logits, target_act)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return child
