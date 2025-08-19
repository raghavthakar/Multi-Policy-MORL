from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import numpy as np

from catserl.shared.actors import Actor

class Checkpoint:
    """
    Minimal, foolproof checkpoint for the *merged* islands.

    Saves only what Stage 2 needs:
      • GA population (as GeneticActor with weights loaded from flat tensors)
      • critics per island (entire modules moved to CPU for portability)
      • island weight vectors
      • a small meta blob (seed, cfg snapshot)

    Usage
    -----
    ckpt = Checkpoint(path)
    ckpt.save_merged(pop, critics_dict, weights_by_island, cfg, seed)

    pop, critics_dict, weights_by_island, meta = ckpt.load_merged(device)
    """

    VERSION = 1

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------- SAVE -------------------------- #
    @torch.no_grad()
    def save_merged(
        self,
        population: List["Actor"],
        critics_by_island: Dict[int, torch.nn.Module],
        weights_by_island: Dict[int, np.ndarray],
        cfg: Dict[str, Any],
        seed: int,
    ) -> None:
        actors_blob: List[Dict[str, Any]] = []
        for actor in population:
            flat = actor.flat_params().detach().cpu().clone()
            actors_blob.append(
                {
                    "kind": str(actor.kind),
                    "pop_id": int(actor.pop_id) if actor.pop_id is not None else -1,
                    "obs_shape": tuple(int(x) for x in actor.impl.obs_shape),
                    "n_actions": int(actor.impl.n_actions),
                    "hidden_dim": int(actor.impl.hidden_dim),
                    "flat": flat,
                }
            )

        critics_blob: Dict[str, Any] = {}
        for island_id, critic in critics_by_island.items():
            # Move to CPU for portability; store whole module for plug‑and‑play use.
            critic_cpu = critic.to("cpu")
            critics_blob[str(int(island_id))] = critic_cpu

        weights_blob = {str(int(k)): np.asarray(v, dtype=np.float32).tolist() for k, v in weights_by_island.items()}

        payload = {
            "version": self.VERSION,
            "meta": {
                "seed": int(seed),
                "cfg": cfg,
            },
            "actors": actors_blob,
            "critics": critics_blob,
            "weights": weights_blob,
        }

        # Single-file atomic-ish save
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.path)

    # -------------------------- LOAD -------------------------- #
    @torch.no_grad()
    def load_merged(
        self,
        device: torch.device | str = "cpu",
        *,
        path: str | Path | None = None,   # <-- explicit override
    ):
        device = torch.device(device)
        load_path = Path(path) if path is not None else self.path
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        # PyTorch 2.6 defaults to weights_only=True; we need full Modules.
        try:
            payload = torch.load(load_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(load_path, map_location="cpu")

        if int(payload.get("version", 0)) != self.VERSION:
            raise RuntimeError(f"Checkpoint version mismatch: found {payload.get('version')}, expected {self.VERSION}")

        from catserl.shared.actors import Actor
        pop = []
        for a in payload["actors"]:
            actor = Actor(
                kind=str(a['kind']),
                pop_id=int(a["pop_id"]),
                obs_shape=tuple(a["obs_shape"]),
                n_actions=int(a["n_actions"]),
                hidden_dim=int(a["hidden_dim"]),
                device=device,
            )
            flat = a["flat"].to(device)
            actor.load_flat_params(flat)
            pop.append(actor)

        critics_dict = {int(k): v.to(device) for k, v in payload["critics"].items()}
        weights_by_island = {int(k): np.asarray(v, dtype=np.float32) for k, v in payload["weights"].items()}
        meta = payload.get("meta", {})
        return pop, critics_dict, weights_by_island, meta