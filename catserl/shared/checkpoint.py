# catserl/shared/checkpoint.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import re

import torch
import numpy as np
import csv

from catserl.shared.actors import Actor
from catserl.shared.buffers import ReplayBuffer
from catserl.shared.rl import TD3
from catserl.shared import data_loader

class Checkpoint:
    """Handles saving and loading of training states."""

    VERSION = 2
    MERGED_POPS_FILENAME = 'merged_populations.dat'
    SNAPSHOT_FILENAME_TEMPLATE = 'island_{island_id}_t{timestep}.ckpt'

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.path / self.MERGED_POPS_FILENAME
        self._merged_stem = Path(self.MERGED_POPS_FILENAME).stem
        self._merged_suffix = Path(self.MERGED_POPS_FILENAME).suffix

    @torch.no_grad()
    def save_island_snapshot(
        self,
        manager_state: Dict,
        island_id: int,
        algorithm: str,
        agent: TD3,
        buffer: ReplayBuffer,
        population: Optional[List[Actor]] = None
    ) -> None:
        """
        Saves a complete snapshot of an island's training state:
          - TD3 agent (nets + optimizers) via agent.save_state()
          - TD3 large replay buffer contents
          - manager_state (timers/counters, etc.)
          - If algorithm == 'pderl': GA population (actors + mini-buffers)
        """
        timestep = manager_state.get('trained_timesteps', 0)

        # ---- Serialize the large TD3 replay buffer ----
        transitions = list(buffer._storage)
        if not transitions:
            states, actions, rewards, next_states, dones = [], [], [], [], []
        else:
            states, actions, rewards, next_states, dones = map(np.stack, zip(*transitions))

        buffer_state = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

        payload: Dict[str, Any] = {
            "version": self.VERSION,
            "algorithm": algorithm,
            "agent_state": agent.save_state(),
            "buffer_state": buffer_state,
            "manager_state": manager_state,
        }

        # ---- If PDERL, serialize GA population with full metadata ----
        if algorithm == 'pderl' and population is not None:
            population_state: List[Dict[str, Any]] = []
            for actor in population:
                # MiniBuffer snapshot; may be None if never initialized
                buf_state = None
                if hasattr(actor, "buffer") and hasattr(actor.buffer, "get_state"):
                    buf_state = actor.buffer.get_state()

                actor_state = {
                    # construction metadata required to rehydrate Actor
                    "kind": actor.kind,
                    "pop_id": actor.pop_id,
                    "obs_shape": actor.obs_shape,
                    "action_type": actor.action_type,
                    "action_dim": actor.action_dim,
                    "hidden_dim": actor.hidden_dim,
                    "max_action": actor.max_action,
                    # MiniBuffer capacity for constructor
                    "buffer_size": getattr(actor.buffer, "max_steps", 0),
                    # parameters + mini-buffer contents
                    "flat_params": actor.flat_params().cpu(),
                    "buffer_state": buf_state,  # None or dict with arrays/ptr/full
                }
                population_state.append(actor_state)

            payload["population_state"] = population_state

        # ---- Atomic write ----
        filename = self.SNAPSHOT_FILENAME_TEMPLATE.format(island_id=island_id, timestep=timestep)
        filepath = self.path / filename
        tmp_path = filepath.with_name(filepath.name + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.rename(filepath)
        print(f"[Checkpoint] Saved island {island_id} snapshot at timestep {timestep}.")

    @torch.no_grad()
    def load_latest_island_snapshot(
        self,
        island_id: int,
        agent: TD3,
        buffer: ReplayBuffer
    ) -> Optional[Dict]:
        """
        Loads the latest snapshot for an island.
        - Always restores TD3 agent + large replay buffer in-place.
        - If snapshot is PDERL, reconstructs GA population and returns it
          embedded in the returned manager_state under 'population'.
        Returns:
            manager_state dict, possibly augmented with 'population' (PDERL).
            None if no snapshot was found.
        """
        pattern = re.compile(f"island_{island_id}_t(\\d+)\\.ckpt$")
        matches: Dict[int, Path] = {}
        for p in self.path.iterdir():
            if not p.is_file():
                continue
            m = pattern.search(p.name)
            if m:
                matches[int(m.group(1))] = p

        if not matches:
            print(f"[Checkpoint] No snapshot found for island {island_id}.")
            return None

        latest_timestep = max(matches.keys())
        latest_ckpt_path = matches[latest_timestep]
        print(f"[Checkpoint] Loading latest snapshot for island {island_id} from: {latest_ckpt_path}")

        payload = torch.load(latest_ckpt_path, map_location=agent.device, weights_only=False)

        # ---- Restore TD3 agent + large buffer ----
        agent.load_state(payload['agent_state'])

        buffer_state = payload['buffer_state']
        buffer._storage.clear()
        for i in range(len(buffer_state['states'])):
            buffer.push(
                buffer_state['states'][i],
                buffer_state['actions'][i],
                buffer_state['rewards'][i],
                buffer_state['next_states'][i],
                buffer_state['dones'][i],
            )

        manager_state: Dict[str, Any] = payload.get('manager_state', {}) or {}

        # ---- If snapshot is for PDERL, rebuild GA population ----
        if payload.get("algorithm") == "pderl" and "population_state" in payload:
            pop: List[Actor] = []
            for a in payload["population_state"]:
                actor = Actor(
                    kind=a.get("kind", "td3"),
                    pop_id=a.get("pop_id", 0),
                    obs_shape=a.get("obs_shape"),
                    action_type=a.get("action_type", "continuous"),
                    action_dim=a.get("action_dim"),
                    hidden_dim=a.get("hidden_dim", 256),
                    max_action=a.get("max_action", 1.0),
                    buffer_size=a.get("buffer_size", 0),  # MiniBuffer max_steps
                    device=agent.device,
                )

                # Load flat params (ensure correct device)
                flat_params = a["flat_params"]
                if not isinstance(flat_params, torch.Tensor):
                    flat_params = torch.tensor(flat_params)
                actor.load_flat_params(flat_params.to(agent.device))

                # Restore MiniBuffer via the official API
                buf_state = a.get("buffer_state", None)
                if hasattr(actor.buffer, "load_state"):
                    actor.buffer.load_state(buf_state)
                else:
                    # Fallback: if no API, clear to empty
                    if hasattr(actor.buffer, "clear"):
                        actor.buffer.clear()

                pop.append(actor)

            # Attach reconstructed population to manager_state
            manager_state = dict(manager_state)
            manager_state["population"] = pop

        return manager_state

    @torch.no_grad()
    def save_merged(
        self,
        population: List[Actor],
        critics_by_island: Dict[int, torch.nn.Module],
        buffers_by_island: Dict[int, Any],
        weights_by_island: Dict[int, np.ndarray],
        cfg: Dict[str, Any],
        seed: int,
        timestep: Union[int, List[int], np.ndarray] = 0
    ) -> None:
        """Saves the necessary components to a single file.

        Each save creates a new file named:
            <MERGED_STEM>_t{TIMESTEP}{SUFFIX}
        e.g. merged_populations_t50000.dat

        The 'timestep' argument may be an int or an iterable of ints (e.g. [t0, t1]).
        If an iterable is provided, the function uses int(sum(timestep)).
        """
        # Normalize timestep (allow list/tuple/np.ndarray as passed by orchestrator)
        if isinstance(timestep, (list, tuple, np.ndarray)):
            ts = int(np.sum(timestep))
        else:
            ts = int(timestep)

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

        critics_blob = {str(k): [c.cpu() for c in v] for k, v in critics_by_island.items()}
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
                # --- Save metadata required for reconstruction ---
                "obs_shape": buffer.obs_shape,
                "action_type": buffer.action_type,
                "action_dim": buffer.action_dim,
            }

        payload = {
            "version": self.VERSION,
            "meta": {"seed": seed, "cfg": cfg, "timestep": ts},
            "actors": actors_blob,
            "critics": critics_blob,
            "weights": weights_blob,
            "island_buffers": buffers_blob,
        }

        # Construct filename: <stem>_t{ts}{suffix}
        final_name = f"{self._merged_stem}_t{ts}{self._merged_suffix}"
        final_path = self.path / final_name
        tmp_path = final_path.with_name(final_path.name + ".tmp")

        # Atomic write
        torch.save(payload, tmp_path)
        tmp_path.replace(final_path)

        # Optionally (backwards compatibility) update the legacy fixed filename to point to the latest save.
        # Write a copy to self.file_path so older loaders that expect merged_populations.dat will still work.
        try:
            torch.save(payload, self.file_path.with_name(self.file_path.name))
        except Exception:
            # Non-fatal: we don't want the checkpoint saving to fail if this secondary write fails.
            pass

    @torch.no_grad()
    def _load_merged(self, device: torch.device | str = "cpu", timestep: Optional[int] = None):
        """Load population, critics, and island buffers saved by save_merged (current format)."""
        if not self.path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.path}")

        def _path_for_ts(ts: int) -> Path:
            return self.path / f"{self._merged_stem}_t{int(ts)}{self._merged_suffix}"

        # Pick file
        if timestep is not None:
            file_to_load = _path_for_ts(int(timestep))
            if not file_to_load.exists():
                raise FileNotFoundError(f"Requested merged checkpoint not found: {file_to_load}")
        else:
            pattern = re.compile(re.escape(self._merged_stem) + r"_t(\d+)" + re.escape(self._merged_suffix) + r"$")
            matches: Dict[int, Path] = {}
            for p in self.path.iterdir():
                if p.is_file():
                    m = pattern.search(p.name)
                    if m:
                        matches[int(m.group(1))] = p
            if not matches:
                # fall back to legacy fixed name only if present; otherwise fail
                if self.file_path.exists():
                    file_to_load = self.file_path
                else:
                    raise FileNotFoundError(f"No merged checkpoint files found in: {self.path}")
            else:
                file_to_load = matches[max(matches.keys())]

        payload = torch.load(file_to_load, map_location="cpu", weights_only=False)
        print(f"[Checkpoint] Loaded data file: {file_to_load}")

        # --- Actors + MiniBuffers (current format) ---
        population: List[Actor] = []
        for ad in payload["actors"]:
            actor = Actor(
                kind=ad["kind"],
                pop_id=ad["pop_id"],
                obs_shape=ad["obs_shape"],
                action_type=ad["action_type"],
                action_dim=ad["action_dim"],
                hidden_dim=ad["hidden_dim"],
                max_action=ad.get("max_action", 1.0),
                buffer_size=ad["buffer"]["max_steps"],
                device=device,
            )

            flat = ad["flat"]
            if not isinstance(flat, torch.Tensor):
                flat = torch.tensor(flat)
            actor.load_flat_params(flat.to(device))

            # ---- MiniBuffer restore (derive 'full' if missing) ----
            buf = ad["buffer"] or {}
            # Compute 'full' robustly from arrays + ptr:
            # If any valid-looking content exists in indices [ptr:max_steps), we assume wraparound happened → full=True.
            ptr = int(buf.get("ptr", 0))
            max_steps = int(buf.get("max_steps", actor.buffer.max_steps))
            states = buf.get("states", None)

            # Heuristic to detect wraparound: look for any non-zero in the tail.
            inferred_full = False
            if isinstance(states, np.ndarray) and max_steps > 0:
                tail = states[ptr:max_steps]
                # any axis nonzero -> filled beyond ptr
                inferred_full = np.any(tail != 0)

            state_dict = {
                "states": buf["states"],
                "actions": buf["actions"],
                "rewards": buf["rewards"],
                "next_states": buf["next_states"],
                "dones": buf["dones"],
                "ptr": ptr,
                "full": bool(buf.get("full", inferred_full)),
            }
            actor.buffer.load_state(state_dict)

            population.append(actor)

        # --- Island ReplayBuffers (large) ---
        buffers_by_island: Dict[int, ReplayBuffer] = {}
        for island_id_str, bd in payload["island_buffers"].items():
            rb = ReplayBuffer(
                obs_shape=bd["obs_shape"],
                action_type=bd["action_type"],
                action_dim=bd["action_dim"],
                capacity=int(bd["capacity"]),
                device=device if isinstance(device, torch.device) else torch.device(device),
            )
            S, A, R, S2, D = bd["states"], bd["actions"], bd["rewards"], bd["next_states"], bd["dones"]
            for i in range(len(S)):
                rb.push(S[i], A[i], R[i], S2[i], D[i])
            buffers_by_island[int(island_id_str)] = rb

        # --- Critics / weights / meta ---
        critics = {int(k): v.to(device) for k, v in payload["critics"].items()}
        weights = {int(k): np.array(v) for k, v in payload["weights"].items()}
        meta = payload.get("meta", {})

        return population, critics, buffers_by_island, weights, meta

    def load_checkpoint(self, device: torch.device):
        try:
            self.loaded_mopderl_slow = False
            self.loaded_catserl_fast = True
            return self._load_merged(device=device)
        except:
            try:
                return data_loader._load_merged_mopderl(root_dir=self.path, merged_stem=self._merged_stem, merged_suffix=self._merged_suffix, device=device)
            except:
                self.loaded_mopderl_slow = True
                self.loaded_catserl_fast = False
                return data_loader._load_mopderl_data(root_dir=self.path, device=device)


    def log_stats(
        self,
        island_id: int,
        generation_number: int,
        cumulative_frames: int,
        vector_return: np.ndarray
    ) -> None:
        """
        Append stats to island_{island_id}_stats.csv with two columns:
          - generation (int)
          - vector_return (string: space-separated numbers in one cell)

        Example cell: "[123.4 -56.7 0.001]"
        """
        # Ensure 1D numpy array of floats
        if isinstance(vector_return, torch.Tensor):
            vec = vector_return.detach().cpu().numpy().ravel().astype(float)
        else:
            vec = np.asarray(vector_return, dtype=float).ravel()

        # Single-cell text for the vector (no commas → no CSV quoting headaches)
        vec_str = np.array2string(vec, separator=' ', max_line_width=10**9)

        stats_path = self.path / f"island_{island_id}_stats.csv"
        file_exists = stats_path.exists()
        mode = "a" if file_exists else "w"

        with stats_path.open(mode, newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generation", "cumulative_frames", "vector_return"])
            writer.writerow([int(generation_number), cumulative_frames, vec_str])
    
    
    def log_pareto_stats(
        self,
        generation_number: int,
        vector_returns: List[np.ndarray] | np.ndarray
    ) -> None:
        """
        Append Pareto-front statistics to pareto_stats.csv.
        
        Columns:
        - generation (int)
        - vector_returns (string): a list where each element is a vector return,
            formatted without commas, e.g.:
            "[[12.3  -4.1]
                [10.2 -10.7]
                [ 3.2  -25.1]]"
        """

        # Convert list of returns into a clean 2D numpy array
        # (each element is a 1D return vector)
        vec_array = []
        for vr in vector_returns:
            if isinstance(vr, torch.Tensor):
                vr = vr.detach().cpu().numpy()
            vr = np.asarray(vr, dtype=float).ravel()
            vec_array.append(vr)
        vec_array = np.stack(vec_array, axis=0)

        # Convert to string with space-separated floats, no commas
        vec_str = np.array2string(
            vec_array,
            separator=' ',
            max_line_width=10**9
        )

        # Build file path
        stats_path = self.path / "pareto_stats.csv"
        file_exists = stats_path.exists()
        mode = "a" if file_exists else "w"

        # Write CSV
        with stats_path.open(mode, newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["generation", "vector_returns"])
            writer.writerow([int(generation_number), vec_str])
