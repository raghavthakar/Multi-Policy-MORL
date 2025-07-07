# catserl/cli/rl_test.py
"""
CLI smoke‑test that wires:
   FourRoomWrapper  +  RLWorker  +  EnvRunner

Run:
   python -m catserl.cli.rl_test --episodes 200
"""
from __future__ import annotations
import argparse, pathlib, sys, random, time
import numpy as np, torch, yaml
from collections import deque

recent10 = deque(maxlen=10)

ROOT = pathlib.Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT))

from catserl.envs.four_room import FourRoomWrapper          # noqa: E402
from catserl.rl.dqn import RLWorker                        # noqa: E402
from catserl.runner import EnvRunner                       # noqa: E402


def load_cfg():
    cfg_path = ROOT / "catserl" / "config" / "default.yaml"
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)



def print_callback(info):
    """
    Called by EnvRunner after each episode.
    info["ret_scalar"]   is already the scalarised return.
    """
    recent10.append(info["ret_scalar"])
    mean10 = sum(recent10) / len(recent10)

    print(f"Ep {info['episode']:03d}"
          f" | len {info['ep_len']:3d}"
          f" | R_vec {info['ret_vec']}"
          f" | R_scalar {info['ret_scalar']:.1f}"
          f" | 10-ep mean {mean10:.2f}")


def main(args):
    cfg = load_cfg()
    device = torch.device(cfg.get("device", "cpu"))

    # seeding
    seed = cfg["seed"]
    np.random.seed(seed);  random.seed(seed);  torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # env + agent
    env = FourRoomWrapper(seed=seed)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    scalar_weight = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    agent = RLWorker(obs_shape, n_actions, scalar_weight, cfg["dqn"], device)
    runner = EnvRunner(env, agent, scalar_weight, callbacks=[print_callback])

    runner.run(n_episodes=args.episodes, learn=True)
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=2000)
    args = p.parse_args()
    t0 = time.time()
    main(args)
    print(f"Elapsed: {time.time()-t0:.1f}s")
