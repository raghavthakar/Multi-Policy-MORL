from __future__ import annotations
from typing import Protocol, Callable, List, Dict, Any
import numpy as np
import time


class AgentProtocol(Protocol):
    def act(self, state: np.ndarray) -> int: ...
    def remember(self, *transition): ...
    def update(self): ...


Callback = Callable[[Dict[str, Any]], None]


class EnvRunner:
    """
    Generic episode runner that handles:
        • ε‑greedy acting by delegating to agent.act()
        • storage & learning if agent exposes remember()/update()
        • per‑episode bookkeeping (length, vector reward, scalar return)

    Callbacks receive a dict:
        {
          "episode": int,
          "ep_len":  int,
          "ret_vec": np.ndarray,
          "ret_scalar": float,
          "t_elapsed": float
        }
    """

    def __init__(self, env, agent: AgentProtocol, scalar_weight: np.ndarray,
                 callbacks: List[Callback] | None = None):
        self.env = env
        self.agent = agent
        self.w = scalar_weight
        self.callbacks = callbacks or []

    # ------------------------------------------------------------------ #
    def run(self, n_episodes: int, learn: bool = True) -> List[np.ndarray]:
        """
        Execute n episodes.  If learn=False the runner will skip
        remember()/update() calls (pure evaluation).
        Returns list of per‑episode reward vectors.
        """
        returns: List[np.ndarray] = []
        t0 = time.time()

        for ep in range(1, n_episodes + 1):
            s, _ = self.env.reset()
            done, ep_len = False, 0
            ret_vec = np.zeros_like(self.w, dtype=np.float32)

            while not done:
                a = self.agent.act(s)
                s2, r_vec, done, trunc, _ = self.env.step(a)
                done = done or trunc

                ret_vec += r_vec
                ep_len += 1

                if learn and hasattr(self.agent, "remember"):
                    self.agent.remember(s, a, r_vec, s2, done)
                    if hasattr(self.agent, "update"):
                        self.agent.update()

                s = s2

            # end episode
            returns.append(ret_vec.copy())
            info = dict(episode=ep,
                        ep_len=ep_len,
                        ret_vec=ret_vec,
                        ret_scalar=float((ret_vec * self.w).sum()),
                        t_elapsed=time.time() - t0)
            for cb in self.callbacks:
                cb(info)

        return returns