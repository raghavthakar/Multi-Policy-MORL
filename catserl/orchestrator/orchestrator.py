# catserl/orchestrator/orchestrator.py

"""
CATSERL Orchestrator

Run the training orchestrator. Allows specifying a YAML config via CLI.

Usage examples:
- Default config (relative to package):
    python -m catserl.orchestrator.orchestrator
- Custom config:
    python -m catserl.orchestrator.orchestrator -c /path/to/config.yaml
- Resume directly into Stage 2 from a merged checkpoint:
    python -m catserl.orchestrator.orchestrator --resume-stage2 /path/to/merged.ckpt
"""

from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "shared" / "config" / "default.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CATSERL Orchestrator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--save-merged", type=Path, default=None, help="If set, save merged islands (population + critics) to this checkpoint file at the end of island training.")
    parser.add_argument("--resume-stage2", type=Path, default=None, help="If set, skip island training and load a merged checkpoint (created by --save-merged) to start Stage 2.")

    args = parser.parse_args(argv)

    cfg_path: Path = args.config
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        return 2

    try:
        import yaml
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}
    except ImportError:
        print("Missing dependency: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Failed to load YAML config at {cfg_path}: {e}", file=sys.stderr)
        return 2

    # Heavy/runtime deps after parsing.
    try:
        import numpy as np
        import torch
        import mo_gymnasium as mo_gym
        from catserl.island.island_manager import IslandManager
        from catserl.orchestrator.checkpoint import Checkpoint
        from catserl.moea.mo_manager import MOManager
    except Exception as e:
        print(f"Failed to import runtime dependencies: {e}", file=sys.stderr)
        return 2

    seed = int(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cpu"))

    env = mo_gym.make("mo-mountaincar-timemove-v0")

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # If resuming Stage 2 directly from a checkpoint, skip island training entirely.
    if args.resume_stage2 is not None:
        mo_mgr = MOManager(env, 2, args.resume_stage2, device=device)
        for _ in range(75):
            mo_mgr.evolve()
        print("MOManager Stage 2 run complete (resumed from checkpoint).")
        return 0
    
    # ---------- Stage 1 (islands) ----------
    # Two islands with different objective weights.
    mgr0 = IslandManager(env, 1, np.array([1, 0]), cfg, seed=seed + 1, device=device)
    mgr1 = IslandManager(env, 2, np.array([0, 1]), cfg, seed=seed + 2, device=device)

    generations = 75
    for gen in range(generations):
        mgr0.train_generation()
        mgr1.train_generation()
        print(
            f"Gen {gen+1:02d} | "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_scalar_returns()[-10:]):.2f} "
            f"Obj-0 10-ep mean: {np.mean(mgr0.get_vector_returns()[-10:], axis=0)} "
            f"| Obj-1 10-ep mean: {np.mean(mgr1.get_scalar_returns()[-10:]):.2f}"
        )

    # Merge islands for potential Stage 2.
    pop0, id0, critic0, w0 = mgr0.export_island()
    pop1, id1, critic1, w1 = mgr1.export_island()

    combined_pop = pop0 + pop1
    critics_dict = {id0: critic0, id1: critic1}
    weights_by_island = {id0: w0, id1: w1}

    # Save the merged state for Stage 2.
    if args.save_merged is not None:
        try:
            ckpt = Checkpoint(args.save_merged)
            ckpt.save_merged(combined_pop, critics_dict, weights_by_island, cfg, seed)
            print(f"Saved merged checkpoint to: {args.save_merged}")
            print("You can now run Stage 2 directly with:")
            print(f"  python -m catserl.orchestrator.orchestrator --resume-stage2 {args.save_merged}")
        except Exception as e:
            print(f"WARNING: Failed to save merged checkpoint: {e}", file=sys.stderr)

    print("Stage 1 complete; merged population prepared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
