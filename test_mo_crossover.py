"""
Unit-test for mo_distilled_crossover using a dummy critic.

Run:
    python test_mo_crossover.py
Expect:
    dominance guard OK
    weights cloned OK
    CE loss X.XXX -> Y.YYY  --> OK
"""

import torch
import numpy as np
import torch.nn.functional as F

from catserl.pderl.population import GeneticActor
from catserl.shared.evo_utils.crossover import mo_distilled_crossover
from catserl.shared.utils.seeding import seed_everything


class DummyCritic(torch.nn.Module):
    """Simple linear critic: maps state to action-value logits."""
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = torch.nn.Linear(obs_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_actor(vec_return, seed):
    """Create GeneticActor, fill its buffer with random states."""
    seed_everything(seed)
    actor = GeneticActor(obs_shape=(4,), n_actions=3)
    actor.vector_return = np.asarray(vec_return, dtype=float)

    # populate MiniBuffer with random (state, action) pairs
    for _ in range(512):
        s = np.random.randn(4).astype(np.float32)
        a = np.random.randint(0, 3)
        actor.remember(s, a)
    return actor


def ce_loss(policy: GeneticActor, critic: torch.nn.Module, batch=128):
    """Compute CE loss of policy vs. critic’s greedy targets on buffer states."""
    # sample states from policy's buffer (we just need a buffer; either parent works)
    states, _ = policy.buffer.sample(batch, torch.device("cpu"))
    with torch.no_grad():
        tgt = critic(states).argmax(dim=1)
    logits = policy.net(states)
    return F.cross_entropy(logits, tgt).item()


if __name__ == "__main__":
    cfg = dict(crossover_lr=1e-3, bc_epochs=1, bc_batch=256)

    # Create two parents where pa Pareto-dominates pb
    pa = make_actor([3, 2, 1], seed=1)
    pb = make_actor([1, 1, 0], seed=2)

    # Dummy critic: same obs_dim & action count
    critic = DummyCritic(obs_dim=4, n_actions=3)

    # 1. Dominance guard: wrong order should raise
    try:
        mo_distilled_crossover(pb, pa, critic, cfg, device="cpu")
    except ValueError:
        print("dominance guard OK")

    # 2. Correct call: better=pa, worse=pb
    child = mo_distilled_crossover(pa, pb, critic, cfg, device="cpu")
    same = torch.allclose(pa.flat_params(), child.flat_params(), atol=1e-6)
    print("weights cloned OK" if same else "weights differ ✗")

    # 3. BC loss should decrease
    before = ce_loss(pa, critic)
    after  = ce_loss(child, critic)
    assert after < before, "CE loss did not decrease!"
    print(f"CE loss {before:.3f} -> {after:.3f}  --> OK")
