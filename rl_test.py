"""
rl_tester.py
Minimal smoke-test for the RLWorker + ReplayBuffer on FourRooms-v0.

Assumes:
  • data.buffers.ReplayBuffer  (imported by rl.dqn)
  • rl.dqn.RLWorker            (code provided earlier)
  • config/default.yaml        (sample shown previously)

No evolution code is touched.
"""
from __future__ import annotations
import argparse
import pathlib
import sys
import time

import gymnasium as gym
import mo_gymnasium
import numpy as np
import torch
import yaml

# --- repo-local imports -----------------------------------------------------#
ROOT = pathlib.Path(__file__).resolve().parents[0]   # adjust if needed
sys.path.append(str(ROOT))

from catserl.rl.dqn import RLWorker         # noqa: E402

# --------------------------------------------------------------------------- #
# helper
# --------------------------------------------------------------------------- #
def load_cfg(path: pathlib.Path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main(args):
    # -------------------- load config ------------------------------------- #
    cfg = load_cfg(ROOT / "catserl" / "config" / "default.yaml")
    device = torch.device(cfg.get("device", "cpu"))

    # -------------------- env --------------------------------------------- #
    env = mo_gymnasium.make("four-room-v0")
    obs_shape = env.observation_space.shape       # e.g. (13,)
    n_actions = env.action_space.n                # 4

    # -------------------- RL worker --------------------------------------- #
    scalar_weight = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    worker = RLWorker(obs_shape,
                      n_actions,
                      scalar_weight,
                      cfg["dqn"],
                      device)

    # -------------------- training loop ----------------------------------- #
    N_EPISODES = args.episodes
    returns_ext = []      # external scalarised return
    returns_vec = []      # vector returns

    for ep in range(1, N_EPISODES + 1):
        s, _ = env.reset(seed=cfg["seed"])
        done, ep_len = False, 0
        ret_vec = np.zeros_like(scalar_weight, dtype=np.float32)

        while not done:
            a = worker.act(s)
            s2, r_vec, done, trunc, _ = env.step(a)
            done = done or trunc

            # bookkeeping
            ret_vec += r_vec
            ep_len += 1

            # store + learn
            worker.remember(s, a, r_vec, s2, done)
            worker.update()

            s = s2

        # end episode
        returns_vec.append(ret_vec)
        returns_ext.append(float(ret_vec[0]))     # scalarised by [1,0,0]

        avg_scalar = np.mean(returns_ext[-10:])
        print(f"Ep {ep:03d} | len {ep_len:3d} | "
              f"R_vec {ret_vec} | R_scalar {ret_vec[0]:.1f} | "
              f"10-ep mean {avg_scalar:.2f}")

    # summary
    print("\nFinished.",
          f"Mean scalar return over {N_EPISODES} eps:",
          f"{np.mean(returns_ext):.2f}")
    env.close()


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3000,
                        help="number of episodes to run")
    args = parser.parse_args()

    tic = time.time()
    main(args)
    print(f"Elapsed: {time.time() - tic:.1f}s")
