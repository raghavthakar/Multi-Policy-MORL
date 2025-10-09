# catserl/shared/evo_utils/eval_pop.py
"""
Population evaluation that computes vector returns and optional transition storage
for per-actor genetic memories. Fitness is not assigned here; callers may
scalarize via their weight vector.
"""
from __future__ import annotations
from typing import List
import numpy as np

from catserl.shared.actors import Actor
from catserl.shared.rollout import deterministic_rollout


def eval_pop(
    pop: List[Actor],
    env,
    weight_vector: np.ndarray,
    episodes_per_actor: int = 1,
    max_ep_len: int = -1,
    rl_worker=None,
    seed: int | None = 2024,
    store_transitions: bool = False,
) -> int:
    """
    Evaluates each actor over a fixed number of episodes, computes the mean
    vector return, and optionally stores transitions into the actor's
    personal buffer. Returns the number of frames collected.
    """
    w = weight_vector
    frames_collected = 0

    for actor in pop:
        vec_return = np.zeros_like(w, dtype=np.float32)
        for ep_num in range(episodes_per_actor):
            ret_vec, ep_len = deterministic_rollout(
                env,
                actor,
                store_transitions=store_transitions,
                max_ep_len=max_ep_len,
                other_actor=rl_worker,
                seed=(seed + ep_num) if seed is not None else None,
            )
            vec_return += ret_vec
            frames_collected += ep_len

        actor.vector_return = vec_return / episodes_per_actor
        print("Actor return: ", actor.vector_return)
    
    return frames_collected