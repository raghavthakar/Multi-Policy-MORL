# checkpoint.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

from catserl.shared.actors import Actor

class Checkpoint:
    """Handles saving and loading the merged population and critics for Stage 2."""

    VERSION = 1

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def save_merged(
        self,
        population: List[Actor],
        critics_by_island: Dict[int, torch.nn.Module],
        weights_by_island: Dict[int, np.ndarray],
        cfg: Dict[str, Any],
        seed: int,
    ) -> None:
        """Saves the necessary components to a single file."""
        actors_blob = []
        for actor in population:
            actors_blob.append({
                # Actor configuration
                "kind": actor.kind,
                "pop_id": actor.pop_id,
                "obs_shape": actor.obs_shape,
                "n_actions": actor.n_actions,
                # FIX: Use the new public property instead of _impl
                "hidden_dim": actor.hidden_dim,
                # Actor state
                "flat": actor.flat_params().cpu(),
                "buffer": {
                    "states": actor.buffer.states,
                    "actions": actor.buffer.actions,
                    "rewards": actor.buffer.rewards,
                    "next_states": actor.buffer.next_states,
                    "dones": actor.buffer.dones,
                    "ptr": actor.buffer.ptr,
                    "max_steps": actor.buffer.max_steps,
                    "max_steps": actor.buffer.max_steps,
                },
            })

        critics_blob = {str(k): v.cpu() for k, v in critics_by_island.items()}
        weights_blob = {str(k): v.tolist() for k, v in weights_by_island.items()}

        payload = {
            "version": self.VERSION,
            "meta": {"seed": seed, "cfg": cfg},
            "actors": actors_blob,
            "critics": critics_blob,
            "weights": weights_blob,
        }

        # Use a temporary file for a safer save operation
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.path)

    @torch.no_grad()
    def load_merged(self, device: torch.device | str = "cpu", path: Path | str = None):
        """Loads and reconstructs the population and critics."""
        self.path = path
        
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.path}")

        payload = torch.load(self.path, map_location="cpu", weights_only=False)

        if payload.get("version") != self.VERSION:
            raise RuntimeError(f"Checkpoint version mismatch: expected {self.VERSION}")

        population = []
        for actor_data in payload["actors"]:
            actor = Actor(
                kind=actor_data['kind'],
                pop_id=actor_data["pop_id"],
                obs_shape=actor_data["obs_shape"],
                n_actions=actor_data["n_actions"],
                hidden_dim=actor_data["hidden_dim"],
                buffer_size=actor_data["buffer"]["max_steps"],
                device=device,
            )
            actor.load_flat_params(actor_data["flat"].to(device))

            # Restore buffer state
            buf_data = actor_data["buffer"]
            actor.buffer.states = buf_data["states"]
            actor.buffer.actions = buf_data["actions"]
            actor.buffer.rewards = buf_data["rewards"]
            actor.buffer.next_states = buf_data["next_states"]
            actor.buffer.dones = buf_data["dones"]
            actor.buffer.ptr = buf_data["ptr"]
            actor.buffer.max_steps = buf_data["max_steps"]
            
            population.append(actor)

        critics = {int(k): v.to(device) for k, v in payload["critics"].items()}
        weights = {int(k): np.array(v) for k, v in payload["weights"].items()}
        meta = payload.get("meta", {})

        return population, critics, weights, meta