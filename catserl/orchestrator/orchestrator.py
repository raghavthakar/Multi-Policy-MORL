# catserl/orchestrator/orchestrator.py

from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "shared" / "config" / "default.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CATSERL Orchestrator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML config file.")
    parser.add_argument("--save-data-dir", type=Path, default=None, help="If set, save checkpoint files to this folder.")
    parser.add_argument("--resume-stage2", action='store_true', default=False, help="If set, skip island training and load a merged checkpoint (from --save-data-dir) to start Stage 2.")

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
        from catserl.shared.checkpoint import Checkpoint
        from catserl.moea.mo_manager import MOManager
    except Exception as e:
        print(f"Failed to import runtime dependencies: {e}", file=sys.stderr)
        return 2

    seed = int(cfg.get("seed", 2024))
    device = torch.device(cfg.get("device", "cpu"))

    env1 = mo_gym.make(cfg['env']['name'])
    env2 = mo_gym.make(cfg['env']['name'])

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    env1.reset(seed=seed)
    env2.reset(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set up the checkpointer
    ckpt=None
    if args.save_data_dir is not None:
        ckpt = Checkpoint(args.save_data_dir)

    # ---------- Stage 1 (islands) ----------
    if not args.resume_stage2:
        # Must first train objective experts on islands.
        # Two islands with different objective weights.
        mgr0 = IslandManager(env1, 1, np.array([1, 0]), cfg, checkpointer=ckpt, seed=seed + 1, device=device)
        mgr1 = IslandManager(env2, 2, np.array([0, 1]), cfg, checkpointer=ckpt, seed=seed + 2, device=device)

        t = [0, 0] # track the trained timeseteps on each objective
        save_merged_pops_every = 10000
        num_checkpts = 0 # track how many times merged populations have been checkpointed
        total_timesteps = cfg['rl']['total_timesteps']
        
        # iterate over training steps
        while sum(t) < total_timesteps:
            t = [mgr0.train(1000), mgr1.train(1000)]
            
            if int(sum(t) / save_merged_pops_every) > int(num_checkpts):
                num_checkpts += 1
                # Save the merged state for Stage 2.
                if args.save_data_dir is not None:
                    try:
                        # Merge islands for potential Stage 2.
                        pop0, id0, critic0, buffer0, w0 = mgr0.export_island()
                        pop1, id1, critic1, buffer1, w1 = mgr1.export_island()
                        combined_pop = pop0 + pop1
                        critics_dict = {id0: critic0, id1: critic1}
                        weights_by_island = {id0: w0, id1: w1}
                        buffers_by_island = {id0: buffer0, id1:buffer1}

                        ckpt.save_merged(combined_pop, critics_dict, buffers_by_island, weights_by_island, cfg, seed, timestep=t)
                        print(f"Saved merged checkpoint to: {args.save_data_dir}")
                    except Exception as e:
                        print(f"WARNING: Failed to save merged checkpoint: {e}", file=sys.stderr)

        print("Stage 1 complete; merged population prepared.")
        return 0

    # ---------- Stage 2 (Expert crossover) ----------
    else:
        # If resuming Stage 2 directly from a checkpoint, skip island training entirely.
        if args.resume_stage2:
            mo_mgr = MOManager(env1, cfg, args.save_data_dir, device=device)
            for _ in range(75):
                mo_mgr.evolve()
            print("MOManager Stage 2 run complete (resumed from checkpoint).")
            return 0

if __name__ == "__main__":
    raise SystemExit(main())
