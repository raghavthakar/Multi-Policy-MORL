# checkpoint.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

from catserl.shared.actors import Actor
from catserl.shared.buffers import ReplayBuffer

class Checkpoint:
    """Handles saving and loading the merged population and critics for Stage 2."""

    VERSION = 1
    FILENAME = 'merged_populations.dat'

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.file_path = self.path / self.FILENAME    # actual checkpoint file

    @torch.no_grad()
    def save_merged(
        self,
        population: List[Actor],
        critics_by_island: Dict[int, torch.nn.Module],
        buffers_by_island: Dict[int, Any],
        weights_by_island: Dict[int, np.ndarray],
        cfg: Dict[str, Any],
        seed: int,
    ) -> None:
        """Saves the necessary components to a single file."""
        # Create the save data directory if it does not exist
        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        actors_blob = []
        for actor in population:
            actors_blob.append({
                "kind": actor.kind,
                "pop_id": actor.pop_id,
                "obs_shape": actor.obs_shape,
                "action_type": actor.action_type,
                "action_dim": actor.action_dim,
                "hidden_dim": actor.hidden_dim,
                "max_action": actor.max_action,
                "flat": actor.flat_params().cpu(),
                "buffer": {
                    "states": actor.buffer.states,
                    "actions": actor.buffer.actions,
                    "rewards": actor.buffer.rewards,
                    "next_states": actor.buffer.next_states,
                    "dones": actor.buffer.dones,
                    "ptr": actor.buffer.ptr,
                    "max_steps": actor.buffer.max_steps,
                },
            })

        critics_blob = {str(k): v.cpu() for k, v in critics_by_island.items()}
        weights_blob = {str(k): v.tolist() for k, v in weights_by_island.items()}
        
        buffers_blob = {}
        for island_id, buffer in buffers_by_island.items():
            transitions = list(buffer._storage)
            if not transitions:
                states, actions, rewards, next_states, dones = [], [], [], [], []
            else:
                states, actions, rewards, next_states, dones = map(np.stack, zip(*transitions))

            buffers_blob[str(island_id)] = {
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "next_states": next_states,
                "dones": dones,
                "capacity": buffer.capacity,
                # --- ADDED: Save metadata required for reconstruction ---
                "obs_shape": buffer.obs_shape,
                "action_type": buffer.action_type,
                "action_dim": buffer.action_dim,
            }

        payload = {
            "version": self.VERSION,
            "meta": {"seed": seed, "cfg": cfg},
            "actors": actors_blob,
            "critics": critics_blob,
            "weights": weights_by_island,
            "island_buffers": buffers_blob,
        }

        tmp_path = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.file_path)

    @torch.no_grad()
    def load_merged(self, device: torch.device | str = "cpu"):
        """Loads and reconstructs the population, critics, and island buffers."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.file_path}")

        payload = torch.load(self.file_path, map_location="cpu", weights_only=False)

        if payload.get("version") != self.VERSION:
            raise RuntimeError(f"Checkpoint version mismatch: expected {self.VERSION}")

        # --- Load Actors (unchanged) ---
        population = []
        for actor_data in payload["actors"]:
            actor = Actor(
                kind=actor_data['kind'],
                pop_id=actor_data["pop_id"],
                obs_shape=actor_data["obs_shape"],
                action_type=actor_data["action_type"],
                action_dim=actor_data["action_dim"],
                hidden_dim=actor_data["hidden_dim"],
                max_action=actor_data.get("max_action", 1.0),
                buffer_size=actor_data["buffer"]["max_steps"],
                device=device,
            )
            actor.load_flat_params(actor_data["flat"].to(device))
            # ... (Restore MiniBuffer state) ...
            buf_data = actor_data["buffer"]
            actor.buffer.states = buf_data["states"]
            actor.buffer.actions = buf_data["actions"]
            actor.buffer.rewards = buf_data["rewards"]
            actor.buffer.next_states = buf_data["next_states"]
            actor.buffer.dones = buf_data["dones"]
            actor.buffer.ptr = buf_data["ptr"]
            population.append(actor)

        # --- Load and reconstruct the large island ReplayBuffers ---
        buffers_by_island = {}
        for island_id_str, buffer_data in payload.get('island_buffers', {}).items():
            # 1. Create a new, empty ReplayBuffer with the saved metadata
            new_buffer = ReplayBuffer(
                obs_shape=buffer_data['obs_shape'],
                action_type=buffer_data['action_type'],
                action_dim=buffer_data['action_dim'],
                capacity=buffer_data['capacity'],
                device=device
            )

            # 2. Unpack the saved data arrays
            states = buffer_data['states']
            actions = buffer_data['actions']
            rewards = buffer_data['rewards']
            next_states = buffer_data['next_states']
            dones = buffer_data['dones']

            # 3. Re-populate the buffer by pushing each saved transition
            for i in range(len(states)):
                new_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
            buffers_by_island[int(island_id_str)] = new_buffer

        critics = {int(k): v.to(device) for k, v in payload["critics"].items()}
        weights = {int(k): np.array(v) for k, v in payload["weights"].items()}
        meta = payload.get("meta", {})

        # --- MODIFIED: Return the loaded buffers ---
        return population, critics, buffers_by_island, weights, meta