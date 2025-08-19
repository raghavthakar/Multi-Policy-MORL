"""
Selection routines for evolutionary algorithms.

- elitist_select: scalar-fitness (μ+λ) selection.
- nondominated_select: NSGA-II Pareto selection (requires pygmo ≥ 2.19).

Both functions expect a list of GeneticActor objects:
    .fitness         -> float     (higher is better)
    .vector_return   -> 1-D np.ndarray  (higher is better)
"""

from __future__ import annotations
from typing import List

import numpy as np
import pygmo as pg  # hard dependency

from catserl.shared.actors import DQNActor  # adjust path if moved


# ------------------------------------------------------------------ #
#  Scalar fitness (Stage-1)
# ------------------------------------------------------------------ #
def elitist_select(pop: List[DQNActor], mu: int) -> List[DQNActor]:
    """Keep the top-`mu` individuals by scalar .fitness (descending)."""
    if mu > len(pop):
        raise ValueError(f"mu ({mu}) > population size ({len(pop)})")
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)[:mu]


# ------------------------------------------------------------------ #
#  NSGA-II selection (Stage-2)
# ------------------------------------------------------------------ #
def nondominated_select(pop: List[DQNActor], mu: int) -> List[DQNActor]:
    """
    Select `mu` individuals via NSGA-II (non-dominated sorting + crowding).

    Raises:
        ValueError – if any individual lacks a 1-D vector_return
        ValueError – if mu > len(pop)
    """
    if mu > len(pop):
        raise ValueError(f"mu ({mu}) > population size ({len(pop)})")

    # build objective matrix (negate: pygmo assumes minimisation)
    points = []
    for ind in pop:
        if ind.vector_return is None or ind.vector_return.ndim != 1:
            raise ValueError("Every individual must have a 1-D vector_return")
        points.append(-ind.vector_return)  # maximise → minimise
    points = np.vstack(points)

    # fast non-dominated sorting
    fronts = pg.fast_non_dominated_sorting(points)[0]

    selected_idx: list[int] = []
    for front in fronts:
        # stop when we already have enough individuals
        if len(selected_idx) >= mu:
            break

        # ------------------------------------------------------------------
        # sort this front by descending crowding distance
        if len(front) > 1:
            cd_subset = pg.crowding_distance(points[front])
            order = [front[i] for i in np.argsort(-cd_subset)]
        else:  # single point: distance undefined, keep as-is
            order = front
        # ------------------------------------------------------------------
        if len(selected_idx) + len(order) <= mu:
            selected_idx.extend(order)
        else:
            selected_idx.extend(order[: mu - len(selected_idx)])
            break

    return [pop[i] for i in selected_idx]
