import sys
import os
import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union, Optional

import numpy as np
import torch

from catserl.shared import actors
from catserl.shared.buffers import ReplayBuffer

# Add MOPDERL to system path
mopderl_dir = os.path.abspath("/nfs/stak/users/thakarr/hpc-share/mopderl-env/mopderl-mo-gymnasium")
sys.path.append(mopderl_dir)

from MOPDERL import ddpg, replay_memory

# --------------------------------------------------------------------------
# Private: MOPDERL Loader
# --------------------------------------------------------------------------

def _parse_mopderl_info_txt(info_file: Path) -> SimpleNamespace:
    """
    Parses MOPDERL's info.txt file into a mock 'args' object.
    """
    if not info_file.exists():
        raise FileNotFoundError(f"MOPDERL info.txt not found at: {info_file}")

    raw_text = info_file.read_text()

    # Clean the text: ast.literal_eval cannot parse "device(type='cuda')"
    # We'll replace it with a placeholder (None) since we override
    # the device later anyway.
    cleaned_text = re.sub(r"device\([^)]+\)", "None", raw_text)

    try:
        # Use ast.literal_eval for safe parsing of Python-like structures
        params_dict = ast.literal_eval(cleaned_text)
    except Exception as e:
        print(f"Error parsing MOPDERL info.txt: {e}")
        raise

    # Convert dict to SimpleNamespace for dot-notation access
    # (e.g., args.state_dim), which MOPDERL's classes expect.
    args = SimpleNamespace(**params_dict)
    return args


def _load_mopderl_data(
    root_dir: Path, device: torch.device
) -> Tuple[List[actors.Actor], Dict[int, torch.nn.Module], Dict[int, ReplayBuffer]]:
    """
    Loads and translates data from the MOPDERL checkpoint format.
    """
    print(f"[Stage1Loader] Loading MOPDERL data from: {root_dir}")

    # 1. Load MOPDERL config ('args') from info.txt
    info_file = root_dir / "info.txt"
    args = _parse_mopderl_info_txt(info_file)

    # CRITICAL: Override the device from the file with the one requested
    # by the user. MOPDERL's networks use args.device upon creation.
    args.device = device

    # Initialize the data structures we need to return
    population: List[actors.Actor] = []
    critics: Dict[int, torch.nn.Module] = {}
    buffers: Dict[int, ReplayBuffer] = {}
    weights: Dict[int, List[int]] = {}

    ckpt_dir = root_dir / "checkpoint"
    warm_up_dir = ckpt_dir / "warm_up"
    agents_dir = warm_up_dir / "rl_agents"

    # 2. Loop over each "island" (MOPDERL calls them rl_agents)
    for island_id in range(args.num_rl_agents):
        agent_dir = agents_dir / str(island_id)
        state_dict_file = agent_dir / "state_dicts.pkl"
        buffer_file = agent_dir / "buffer.npy"

        if not state_dict_file.exists():
            print(
                f"Warning: MOPDERL state_dicts.pkl not found for "
                f"island {island_id}. Skipping."
            )
            continue
        if not buffer_file.exists():
            print(
                f"Warning: MOPDERL buffer.npy not found for "
                f"island {island_id}. Skipping."
            )
            continue

        # Load the saved weights
        sd = torch.load(state_dict_file, map_location="cpu")

        # --- 3. Translate Actor ---
        # Instantiate a catserl.Actor wrapper
        # We must use 'kind="td3"' so it's compatible with MOManager
        wrapper_actor = actors.Actor(
            kind="td3",
            pop_id=island_id,
            obs_shape=(args.state_dim,),  # MOPDERL uses flat states
            action_type="continuous",
            action_dim=args.action_dim,
            hidden_dim=args.ls,  # From info.txt
            max_action=1.0,  # MOPDERL actor uses tanh, so max_action is 1.0
            device=device,
            # buffer_size is for the actor's MiniBuffer, can be default
        )

        # Instantiate the *MOPDERL* network
        mopderl_actor_net = ddpg.Actor(args).to(device)
        mopderl_actor_net.load_state_dict(sd['actor'])
        mopderl_actor_net.eval()

        # --- THE HACK ---
        # Overwrite the default policy in the catserl.Actor wrapper
        # with the loaded MOPDERL policy network.
        # This works because all methods (act, flat_params)
        # are routed to `_impl.net`.
        wrapper_actor._impl.net = mopderl_actor_net
        # Also update max_action, as MOPDERL's is 1.0
        wrapper_actor._impl.max_action = 1.0

        population.append(wrapper_actor)

        # --- 4. Translate Critic ---
        # This is simple: just load the MOPDERL critic module.
        mopderl_critic_net = ddpg.Critic(args).to(device)
        mopderl_critic_net.load_state_dict(sd['critic'])
        mopderl_critic_net.eval()
        critics[island_id] = mopderl_critic_net

        # --- 5. Translate Buffer ---
        # Load MOPDERL buffer
        mopderl_buf = replay_memory.ReplayMemory(
            capacity=args.buffer_size,
            device="cpu"  # Load to CPU first
        )
        mopderl_buf.load_info(buffer_file)

        # Create a new catserl.ReplayBuffer
        catserl_buf = ReplayBuffer(
            obs_shape=(args.state_dim,),
            action_type="continuous",
            action_dim=args.action_dim,
            capacity=mopderl_buf.capacity,
            device=device  # Target device
        )

        # Copy transitions one by one, translating format
        for trans in mopderl_buf.memory:
            # MOPDERL stores (1, dim) arrays, squeeze them
            s = trans.state.squeeze(0)
            a = trans.action.squeeze(0)
            r = trans.reward.squeeze(0)
            s2 = trans.next_state.squeeze(0)
            d = bool(trans.done.item())
            
            # Push into our buffer
            catserl_buf.push(s, a, r, s2, d)

        buffers[island_id] = catserl_buf
        weights[island_id] = np.array([0 if i == island_id else 1 for i in range(args.num_rl_agents)]) #HACK

    print(f"[Stage1Loader] Loaded {len(population)} actors, {len(critics)} critics, {len(buffers)} buffers (MOPDERL).")
    return population, critics, buffers, weights, 10


@torch.no_grad()
def _load_merged_mopderl(root_dir: Path, merged_stem: str, merged_suffix: str, device: torch.device | str = "cpu", timestep: Optional[int] = None):
    """
    Fast Loader for Cached MOPDERL Data.
    
    Uses info.txt to strictly reconstruct the MOPDERL architecture
    before loading the cached flat parameters.
    """
    # 1. Verify Root
    if not root_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {root_dir}")

    # 2. Retrieve info.txt args (Ground Truth for Architecture)
    info_file = root_dir / "info.txt"
    if not info_file.exists():
        raise FileNotFoundError(f"Cannot reconstruct MOPDERL architecture: {info_file} is missing.")
    
    # Parse the args that MOPDERL expects (state_dim, ls, use_ln, etc.)
    mopderl_args = _parse_mopderl_info_txt(info_file)
    mopderl_args.device = device  # Enforce correct device

    # 3. Find the cache file
    def _path_for_ts(ts: int) -> Path:
        return root_dir / f"{merged_stem}_t{int(ts)}{merged_suffix}"

    if timestep is not None:
        file_to_load = _path_for_ts(int(timestep))
    else:
        pattern = re.compile(re.escape(merged_stem) + r"_t(\d+)" + re.escape(merged_suffix) + r"$")
        matches: Dict[int, Path] = {}
        for p in root_dir.iterdir():
            if p.is_file():
                m = pattern.search(p.name)
                if m: matches[int(m.group(1))] = p
        if not matches:
            raise FileNotFoundError(f"No merged checkpoint files found in: {root_dir}")
        file_to_load = matches[max(matches.keys())]

    print(f"[Checkpoint] Loading MOPDERL-based cache file: {file_to_load}")
    payload = torch.load(file_to_load, map_location="cpu", weights_only=False)

    # 4. Reconstruct Population
    population: List[actors.Actor] = []
    for ad in payload["actors"]:
        # A. Initialize Standard Wrapper (Shell)
        actor = actors.Actor(
            kind=ad["kind"], pop_id=ad["pop_id"],
            obs_shape=ad["obs_shape"], action_type=ad["action_type"],
            action_dim=ad["action_dim"], hidden_dim=ad["hidden_dim"],
            max_action=ad.get("max_action", 1.0), buffer_size=ad["buffer"]["max_steps"],
            device=device,
        )

        # B. Hydrate MOPDERL Architecture
        # We use `mopderl_args` (from info.txt) to ensure `use_ln` and exact dimensions are respected.
        try:
            mopderl_net = ddpg.Actor(mopderl_args).to(device)
            mopderl_net.eval()
            
            # Perform the swap
            actor._impl.net = mopderl_net
            actor._impl.max_action = 1.0
        except Exception as e:
            print(f"[Checkpoint] Error reconstructing MOPDERL network using info.txt args: {e}")
            raise RuntimeError("Failed to hydrate MOPDERL architecture.")

        # C. Load Parameters
        flat = ad["flat"]
        if not isinstance(flat, torch.Tensor):
            flat = torch.tensor(flat)
        
        # This should now work perfectly as dimensions and flags (LN) match exactly
        actor.load_flat_params(flat.to(device))

        # D. Restore MiniBuffer
        buf = ad["buffer"] or {}
        ptr = int(buf.get("ptr", 0))
        max_steps = int(buf.get("max_steps", actor.buffer.max_steps))
        states = buf.get("states", None)
        inferred_full = False
        if isinstance(states, np.ndarray) and max_steps > 0:
            tail = states[ptr:max_steps]
            inferred_full = np.any(tail != 0)

        state_dict = {
            "states": buf["states"], "actions": buf["actions"],
            "rewards": buf["rewards"], "next_states": buf["next_states"],
            "dones": buf["dones"], "ptr": ptr,
            "full": bool(buf.get("full", inferred_full)),
        }
        actor.buffer.load_state(state_dict)
        population.append(actor)

    # 5. Load Island Buffers
    buffers_by_island: Dict[int, ReplayBuffer] = {}
    for island_id_str, bd in payload["island_buffers"].items():
        rb = ReplayBuffer(
            obs_shape=bd["obs_shape"], action_type=bd["action_type"],
            action_dim=bd["action_dim"], capacity=int(bd["capacity"]),
            device=device if isinstance(device, torch.device) else torch.device(device),
        )
        S, A, R, S2, D = bd["states"], bd["actions"], bd["rewards"], bd["next_states"], bd["dones"]
        for i in range(len(S)):
            rb.push(S[i], A[i], R[i], S2[i], D[i])
        buffers_by_island[int(island_id_str)] = rb

    # 6. Load Critics and Metadata
    critics = {int(k): v.to(device) for k, v in payload["critics"].items()}
    weights = {int(k): np.array(v) for k, v in payload["weights"].items()}
    meta = payload.get("meta", {})

    return population, critics, buffers_by_island, weights, meta