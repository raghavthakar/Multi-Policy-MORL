# catserl/moea/mo_manager.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Generator, Optional

import numpy as np
import torch

from catserl.orchestrator.checkpoint import Checkpoint


__all__ = ["MOManager"]


class MOManager:
    """
    Minimal MO stage manager (Stage 2).

    Current behavior (per your request):
      • __init__: load a *merged* checkpoint (population, critics, island weights, meta)
      • evolve: print a dyadic schedule of scalarisations (no evolution yet)

    Notes
    -----
    - Expect the checkpoint to have been created via Checkpoint.save_merged(...)
    - This is intentionally skeletal; you can flesh out evolve(...) next.
    """
    def __init__(self, ckpt_path: str | Path, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.ckpt_path = Path(ckpt_path).resolve()
        print(f"[MOManager] Loading merged checkpoint from: {self.ckpt_path}")

        ckpt = Checkpoint(self.ckpt_path)
        pop, critics_dict, weights_by_island, meta = ckpt.load_merged(
            device=self.device,
            path=self.ckpt_path,
        )

        self.population = pop
        self.critics = critics_dict
        self.weights_by_island = weights_by_island
        self.meta = meta

        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics.")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def evolve(self, n_points: int = 8, include_endpoints: bool = False) -> None:
        """
        For now: just print a dyadic schedule of target scalarisations.

        Parameters
        ----------
        n_points : int
            Number of dyadic midpoints to produce (not counting endpoints by default).
        include_endpoints : bool
            If True, include 0.0 and 1.0 at the beginning/end of the printed schedule.
        """
        schedule = list(self._dyadic_schedule(n_points, include_endpoints=include_endpoints))
        print(f"[MOManager] Dyadic schedule (n={n_points}, include_endpoints={include_endpoints}):")
        print(schedule)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _dyadic_schedule(n: int, *, include_endpoints: bool = False) -> Generator[float, None, None]:
        """
        Generate the classic dyadic mid-point schedule:
            0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, ...

        By default, endpoints 0.0 and 1.0 are omitted (you already have experts there).

        Parameters
        ----------
        n : int
            How many dyadic values to yield.
        include_endpoints : bool
            Whether to include 0.0 and 1.0 once at the ends.

        Yields
        ------
        float
            Next dyadic scalarisation in (0,1) (or [0,1] if include_endpoints=True).
        """
        if n <= 0:
            if include_endpoints:
                yield 0.0
                yield 1.0
            return

        emitted = 0

        if include_endpoints:
            yield 0.0

        k = 1  # denominator = 2**k
        # Keep emitting until we have 'n' interior points
        while emitted < n:
            denom = 2 ** k
            # odd numerators only: 1,3,5,...,denom-1
            for num in range(1, denom, 2):
                alpha = num / denom
                # Guard against floating weirdness; round for nicer printing
                yield float(round(alpha, 10))
                emitted += 1
                if emitted >= n:
                    break
            k += 1

        if include_endpoints:
            yield 1.0
