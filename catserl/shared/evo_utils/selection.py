# catserl/pderl/selection.py
"""
Selection routines for evolutionary algorithms.
"""
from __future__ import annotations
from typing import List
import numpy as np

from catserl.shared.actors import Actor


def elitist_select(pop: List[Actor], num_elites: int) -> List[Actor]:
    """
    Selects the top-performing individuals from a population based on fitness.
    """
    if num_elites > len(pop) or num_elites < 0:
        raise ValueError(f"Number of elites ({num_elites}) is invalid for population size ({len(pop)}).")
    
    # Sort the population in descending order of fitness and return the top slice.
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)[:num_elites]


def selection_tournament(pop: List[Actor], num_to_select: int, tournament_size: int) -> List[Actor]:
    """
    Selects individuals from a population using a tournament.

    In each tournament, a small, random subset of the population is chosen,
    and the fittest individual from that subset is selected. This process is
    repeated until the desired number of individuals has been selected.
    """
    selected = []
    pop_indices = list(range(len(pop)))

    for _ in range(num_to_select):
        # Select random competitors for the tournament.
        competitor_indices = np.random.choice(pop_indices, size=tournament_size, replace=False)
        
        # Find the competitor with the highest fitness.
        winner = max(competitor_indices, key=lambda i: pop[i].fitness)
        selected.append(pop[winner])
        
    return selected