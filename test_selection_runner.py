"""
Sanity-check the two selection routines using real GeneticActor objects.

Run:
    python test_selection_runner.py
Expect:
    All selection tests: OK
"""

import numpy as np
from catserl.shared.evo_utils import selection
from catserl.pderl.population import GeneticActor


def make_actor(fid, fitness, vec):
    """Create a minimal GeneticActor and patch in fitness / vector info."""
    # 1×1 dummy observation & action space (unused in selection)
    actor = GeneticActor(obs_shape=(1,), n_actions=1)
    actor.id = fid               # optional tag for assertions
    actor.fitness = fitness
    actor.vector_return = np.asarray(vec, dtype=float)
    return actor


def test_elitist():
    pop = [
        make_actor("A", 10.0, [1, 0]),
        make_actor("B", 11.0, [2, 0]),
        make_actor("C", 9.0, [0, 3]),
        make_actor("D", 12.0, [0, 4]),
    ]
    sel = selection.elitist_select(pop, mu=2)
    assert {s.id for s in sel} == {"D", "B"}, "Elitist selection failed"


def test_nsga2():
    pop = [
        make_actor("A", 0, [1, 4]),
        make_actor("B", 0, [2, 3]),
        make_actor("C", 0, [3, 2]),
        make_actor("D", 0, [4, 1]),
        make_actor("E", 0, [0, 0]),  # dominated
    ]
    sel = selection.nondominated_select(pop, mu=4)
    assert {s.id for s in sel} == {"A", "B", "C", "D"}, "NSGA-II selection failed"


if __name__ == "__main__":
    test_elitist()
    test_nsga2()
    print("All selection tests: OK")
