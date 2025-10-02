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
    parser.add_argument("--resume-stage1", action='store_true', default=False, help="If set, resume Stage 1 island training from a checkpoint (from --save-data-dir).")
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
        mgr0 = IslandManager(env1, 1, np.array([1, 0]), list([np.array([0, 1])]), cfg, checkpointer=ckpt, seed=seed + 1, device=device)
        mgr1 = IslandManager(env2, 2, np.array([0, 1]), list([np.array([1, 0])]), cfg, checkpointer=ckpt, seed=seed + 2, device=device)

        # If the resume flag is set, load the state for each manager.
        if args.resume_stage1:
            print("\n--- Resuming Stage 1 Training from Checkpoints ---")
            if ckpt:
                mgr0.resume_from_checkpoint()
                mgr1.resume_from_checkpoint()
            else:
                print("WARNING: --resume-stage1 flag set, but no checkpoint directory provided. Starting from scratch.")
        
        # Initialize timestep tracker *after* potential resume.
        t = [mgr0.trained_timesteps, mgr1.trained_timesteps]
        save_merged_pops_every = 60000
        num_checkpts = sum(t) // save_merged_pops_every
        total_timesteps = cfg['rl']['total_timesteps']
        
        # The training loop now correctly starts from the resumed timestep.
        while t[0] < total_timesteps or t[1] < total_timesteps:
            # Only train managers that haven't finished.
            if mgr0.trained_timesteps < total_timesteps:
                mgr0.train(1000)
            if mgr1.trained_timesteps < total_timesteps:
                mgr1.train(1000)
            
            t = [mgr0.trained_timesteps, mgr1.trained_timesteps]
            
            # Periodically save a merged checkpoint for Stage 2.
            current_total_steps = sum(t)
            if (current_total_steps // save_merged_pops_every) > num_checkpts:
                num_checkpts = (current_total_steps // save_merged_pops_every)
                # Save the merged state for Stage 2.
                if ckpt:
                    try:
                        # Merge islands for potential Stage 2.
                        pop0, id0, critic0, buffer0, w0 = mgr0.export_island()
                        pop1, id1, critic1, buffer1, w1 = mgr1.export_island()
                        ckpt.save_merged(pop0 + pop1, {id0: critic0, id1: critic1}, 
                                         {id0: buffer0, id1:buffer1}, {id0: w0, id1: w1}, cfg, seed, timestep=t)
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
