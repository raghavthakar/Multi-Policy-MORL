# ==================== catserl/rollout.py ====================
"""
Stateless single-episode executor.

A policy must expose:
    act(state)                 -> action (int)
    remember(s,a,r,s2,done)    (optional)
    update()                   (optional)
"""

from __future__ import annotations
from typing import Tuple, Any
import numpy as np


def rollout(env, policy, learn: bool = True) -> Tuple[np.ndarray, int]:
    """
    Runs ONE episode and (optionally) lets the policy learn online.

    Returns
    -------
    ret_vec : np.ndarray   -- episode-summed reward vector
    ep_len  : int          -- steps taken
    """
    s, _ = env.reset()
    done, trunc, ep_len = False, False, 0
    ret_vec = None

    while not (done or trunc):
        a = policy.act(s)
        s2, r_vec, done, trunc, _ = env.step(a)

        if ret_vec is None:
            ret_vec = np.array(r_vec, dtype=np.float32)
        else:
            ret_vec += r_vec

        if learn and hasattr(policy, "remember"):
            policy.remember(s, a, r_vec, s2, done or trunc)
            if hasattr(policy, "update"):
                policy.update()

        s = s2
        ep_len += 1

    return ret_vec, ep_len