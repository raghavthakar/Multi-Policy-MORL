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