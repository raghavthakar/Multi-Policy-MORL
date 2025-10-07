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


def deterministic_rollout(env, actor, store_transitions: bool = True, max_ep_len: int = -1, other_actor=None, seed: int = None) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Runs ONE episode and (optionally) stores transitions.
    
    Parameters
    ----------
    env : environment instance
    actor : Genetic/RLAlgo actor that is being rolled out
    store_transitions : bool, optional
        Whether to store transitions into actor's buffer (default: True)
    max_ep_len : int, optional
        Maximum episode length (default: -1, meaning no limit)
    other_actor : policy instance, optional
        If provided, calls other_actor.remember(s, a, r_vec, s2, done) at each step (default: None)
    seed : int, optional
        If provided, will reset the environment to exactly the seed.

    Returns
    -------
    ret_vec : np.ndarray   -- episode-summed reward vector
    ep_len  : int          -- steps taken
    """
    if seed:
        s, _ = env.reset(seed=seed)
    else:
        s, _ = env.reset()
    done, trunc, ep_len = False, False, 0
    ret_vec = None
    ep_len = 0

    while not (done or trunc):
        a = actor.act(s, noisy_action=False)
        s2, r_vec, done, trunc, info = env.step(a)
        if done == True:
            print("WOOOOO")

        if ret_vec is None:
            ret_vec = np.array(r_vec, dtype=np.float32)
        else:
            ret_vec += r_vec

        if store_transitions and hasattr(actor, "remember"):
            actor.remember(s, a, r_vec, s2, done or trunc)
            if other_actor is not None and hasattr(other_actor, "remember"):
                other_actor.remember(s, a, r_vec, s2, done or trunc)

        s = s2
        ep_len += 1

        if max_ep_len > 0 and ep_len >= max_ep_len:
            trunc = True
            
    return ret_vec, ep_len
