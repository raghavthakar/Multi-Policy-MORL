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


def rollout(env, policy, learn: bool = True, max_ep_len: int = -1) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Runs ONE episode and (optionally) lets the policy learn online.

    Returns
    -------
    ret_vec : np.ndarray   -- episode-summed reward vector
    ep_len  : int          -- steps taken
    ext_ret_vec : np.ndarray -- episode-summed extrinsic reward vector
    """
    s, _ = env.reset()
    done, trunc, ep_len = False, False, 0
    ret_vec = None
    ext_ret_vec = None
    ep_len = 0

    while not (done or trunc):
        a = policy.act(s)
        s2, r_vec, done, trunc, info = env.step(a)

        if ret_vec is None:
            ret_vec = np.array(r_vec, dtype=np.float32)
        else:
            ret_vec += r_vec
        
        if ext_ret_vec is None:
            ext_ret_vec = np.array(info["extrinsic"], dtype=np.float32)
        else:
            ext_ret_vec += info["extrinsic"]

        if learn and hasattr(policy, "remember"):
            policy.remember(s, a, r_vec, s2, done or trunc)
            if hasattr(policy, "update"):
                policy.update()

        s = s2
        ep_len += 1

        if max_ep_len > 0 and ep_len >= max_ep_len:
            trunc = True

    if ep_len != max_ep_len:
        print(f"Warning: Episode ended at {ep_len} steps, not {max_ep_len}.")
    return ret_vec, ep_len, ext_ret_vec