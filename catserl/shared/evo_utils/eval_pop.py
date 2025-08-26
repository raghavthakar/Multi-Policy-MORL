"""
One PDERL 'generation' **without** variation (for now):
  • Evaluate every GeneticActor
  • Update their .fitness fields
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np

from catserl.shared.actors import Actor
from catserl.shared.rollout import deterministic_rollout


def eval_pop(pop: List[Actor],
                env,
                weight_vector: np.ndarray,
                episodes_per_actor: int = 1,
                max_ep_len: int = -1,
                rl_worker=None) -> Dict:
    """
    Returns simple stats; population's fitnesses are modified in-place.

    Parameters
    ----------
    pop : List[GeneticActor]
        Population of genetic actors to evaluate.
    env : environment instance
        The environment to use for evaluation.
    weight_vector : np.ndarray
        Weight vector for scalarizing returns.
    episodes_per_actor : int, optional
        Number of episodes per actor (default: 1).
    max_ep_len : int, optional
        Maximum episode length (default: -1, meaning no limit).
    rl_worker : RLWorker, optional
        If provided, will be supplied as 'other_actor' to rollout so it can observe transitions (default: None).

    Returns
    -------
    stats : dict
        Dictionary with mean, max fitness, and population size.
    frames_collected : int
        Total frames collected during evaluation.
    """
    w = weight_vector
    frames_collected = 0
    seed = 42

    for actor in pop:
        vec_return = np.zeros_like(w, dtype=np.float32)
        for ep_num in range(episodes_per_actor):
            ret_vec, ep_len = deterministic_rollout(
                env, actor, store_transitions=False, max_ep_len=max_ep_len, other_actor=rl_worker, seed=seed+ep_num
            )  # NOTE: Set learn=True to update actor's buffer
            vec_return += ret_vec
            frames_collected += ep_len

        actor.vector_return = vec_return / episodes_per_actor
        print("Deterministic performance: ", actor.vector_return)
