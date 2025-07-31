# =======================================================================
#  catserl/moea/moea.py          →  catserl/mopderl/mo_manager.py
#
#  MOManager  —  Multi-Objective PDERL evolution stage (MOPDERL)
#  ---------------------------------------------------------------------
#  • Receives one merged population of GeneticActor objects.
#  • Runs the MOPDERL loop for a fixed number of generations:
#        evaluate → NSGA-II select → MO-distilled crossover → proximal mutation
# =======================================================================

from __future__ import annotations
from typing import List, Dict

import random
import torch
import pygmo

from catserl.shared.envs.four_room import FourRoomWrapper
from catserl.island.genetic_actor import GeneticActor
from catserl.shared.evo_utils import selection, crossover, proximal_mutation, eval_pop


def dominates(a, b):
    """
    Returns True if vector a Pareto-dominates vector b (assumes maximization).
    Uses pygmo's dominance logic.
    """
    return pygmo.pareto_dominance([-x for x in a], [-x for x in b])

class MOManager:
    """Stage-2 multi-objective evolution over a merged population."""

    # ------------------------------------------------------------------ #
    def __init__(
        self,
        population: List[GeneticActor],
        cfg: Dict,
        device: torch.device | str = "cpu",
    ):
        self.pop: List[GeneticActor] = population
        self.cfg = cfg
        self.device = torch.device(device)

        # Novelty OFF environment
        self.env = FourRoomWrapper(seed=cfg["seed"] + 999, beta=0.0)

        self.max_ep_len = cfg["env"]["max_ep_len"]
        self.episodes_per_actor = cfg["mopderl"]["episodes_per_actor"]

        self.g = 0  # generation counter

    # ------------------------------------------------------------------ #
    def evolve(self, generations: int, critics_dict: Dict[int, torch.nn.Module]) -> None:
        """
        Run the MOPDERL evolution loop.

        Parameters
        ----------
        generations   : int
            Number of MO generations to perform.
        critics_dict  : Dict[int, torch.nn.Module]
            Mapping pop_id → critic frozen at end of Stage-1.
            Each GeneticActor stores its pop_id and must find its critic here.
        """
        for _ in range(generations):
            self._one_generation(critics_dict)
            self.g += 1

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #
    def _one_generation(self, critics: Dict[int, torch.nn.Module]) -> None:
        """One MOPDERL generation (eval → select → crossover → mutate)."""

        # 1. Evaluate population on true objectives
        eval_pop.eval_pop(
            self.pop,
            env=self.env,
            weight_vector=[1, 1, 1],
            episodes_per_actor=self.episodes_per_actor,
            max_ep_len=self.max_ep_len,
        )

        # 2. NSGA-II elite selection  (μ = N/2)
        mu = len(self.pop) // 2
        elite = selection.nondominated_select(self.pop, mu)

        # Sort elites by NSGA-II rank and crowding distance using maximisation (negate objectives)
        negated_points = [[-v for v in x.vector_return] for x in elite]
        sorted_indices = pygmo.sort_population_mo(negated_points)
        elite_to_rank = {elite[idx]: rank for rank, idx in enumerate(sorted_indices)}

        # 3. Produce μ children via MO-distilled crossover
        children: List[GeneticActor] = []
        while len(children) < mu:
            pa, pb = random.sample(elite, 2)

            # Decide better vs worse based on sort_population_mo rank
            rank_pa = elite_to_rank[pa]
            rank_pb = elite_to_rank[pb]
            if rank_pa < rank_pb:
                better, worse = pa, pb
            elif rank_pb < rank_pa:
                better, worse = pb, pa
            else:
                # If ranks are equal, break ties arbitrarily (keep order)
                better, worse = pa, pb

            worse_critic = critics[worse.pop_id]          # fetch critic by id

            child = crossover.mo_distilled_crossover(
                better_parent=better,
                worse_parent=worse,
                critic=worse_critic,
                cfg=self.cfg["pderl"],
                device=self.device,
            )
            # Inherit pop_id = critic id of worse parent (follows paper)
            child.pop_id = worse.pop_id
            children.append(child)

        # 4. Proximal mutation (use any elite critic for Jacobian)
        proximal_mutation.proximal_mutate(
            elite + children,
            critics[elite[0].pop_id],
            # sigma=self.cfg["pderl"]["sigma"],
        )

        # 5. Update population  (size unchanged: μ elites + μ children)
        self.pop = elite + children

        # Light progress print
        if (self.g + 1) % 10 == 0 or self.g == 0:
            print(f"[MOManager] Gen {self.g:04d} completed — pop {len(self.pop)}")
