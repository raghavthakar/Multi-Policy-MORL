import sys
import os
import ast
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from catserl.shared import actors
from catserl.shared.buffers import ReplayBuffer

# Add MOPDERL to system path
mopderl_dir = os.path.abspath("/home/raghav/Research/mopderl-env/mopderl")
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

    print(
        f"[Stage1Loader] Loaded {len(population)} actors, {len(critics)} "
        f"critics, {len(buffers)} buffers (MOPDERL)."
    )

    return population, critics, buffers, 10, 10