# catserl/pderl/selection.py
"""
Selection routines for evolutionary algorithms.
"""
from __future__ import annotations
from typing import List

from catserl.shared.actors import Actor


def elitist_select(pop: List[Actor], num_elites: int) -> List[Actor]:
    """
    Selects the top-performing individuals from a population based on fitness.

    Args:
        pop: A list of actor individuals to be sorted.
        num_elites: The number of top individuals to select.

    Returns:
        A new list containing the `num_elites` actors with the highest fitness.
    """
    if num_elites > len(pop):
        raise ValueError(f"Number of elites ({num_elites}) cannot exceed population size ({len(pop)}).")
    
    # Sort the population in descending order of fitness and return the top slice.
    return sorted(pop, key=lambda ind: ind.fitness, reverse=True)[:num_elites]