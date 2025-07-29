"""
One PDERL 'generation' **without** variation (for now):
  • Evaluate every GeneticActor
  • Update their .fitness fields
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np

from catserl.island.population import GeneticActor
from catserl.shared.evo_utils.selection import elitist_select
from catserl.shared.envs.rollout import rollout


def eval_pop(pop: List[GeneticActor],
                env,
                weight_vector: np.ndarray,
                episodes_per_actor: int = 1,
                max_ep_len: int = -1) -> Dict:
    """
    Returns simple stats; population's fitnesses are modified in-place.
    """
    w = weight_vector
    fitness_vals = []
    frames_collected = 0

    for actor in pop:
        vec_return = np.zeros_like(w, dtype=np.float32)
        for _ in range(episodes_per_actor):
            ret_vec, ep_len, _ = rollout(env, actor, learn=True, max_ep_len=max_ep_len) # NOTE: Set learn=True to update actor's buffer
            vec_return += ret_vec
            frames_collected += ep_len

        actor.vector_return = vec_return / episodes_per_actor
        actor.fitness = float((actor.vector_return * w).sum())
        fitness_vals.append(actor.fitness)

    return dict(mean_fitness=np.mean(fitness_vals),
                max_fitness=np.max(fitness_vals),
                population_size=len(pop)), frames_collected
