"""
Very simple elitist (μ+λ) selection: keep top-`mu` by .fitness (higher = better).
"""
from __future__ import annotations
from typing import List
from catserl.pderl.population import GeneticActor


def elitist_select(pop: List[GeneticActor], mu: int) -> List[GeneticActor]:
    pop_sorted = sorted(pop, key=lambda ind: ind.fitness, reverse=True)
    return pop_sorted[:mu]
