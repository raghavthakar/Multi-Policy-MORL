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
        # Fallback: Create a minimal mock args if file missing
        print(f"[Loader] info.txt not found at {info_file}, using defaults.")
        return SimpleNamespace(
            state_dim=10, # REPLACE WITH DEFAULTS IF NEEDED
            action_dim=2,
            ls=200, # Hidden size
            buffer_size=1000000,
            num_rl_agents=2,
            num_objectives=2,
            device='cpu' 
        )

    raw_text = info_file.read_text()
    # Clean the text: ast.literal_eval cannot parse "device(type='cuda')"
    cleaned_text = re.sub(r"device\([^)]+\)", "'cpu'", raw_text)

    try:
        params_dict = ast.literal_eval(cleaned_text)
    except Exception as e:
        print(f"Error parsing MOPDERL info.txt: {e}")
        raise

    return SimpleNamespace(**params_dict)

def _load_single_actor(
    state_dict_path: Path, 
    args: SimpleNamespace, 
    device: torch.device, 
    pop_id: int,
    is_genetic_agent: bool
) -> actors.Actor:
    """
    Helper to load a single actor (either RL-Teacher or Genetic-Student)
    and wrap it in a catserl.Actor.
    """
    # 1. Init wrapper
    wrapper_actor = actors.Actor(
        kind="td3", # Compatible generic type
        pop_id=pop_id,
        obs_shape=(args.state_dim,),
        action_type="continuous",
        action_dim=args.action_dim,
        hidden_dim=args.ls,
        max_action=1.0,
        device=device,
    )

    # 2. Init MOPDERL network
    mopderl_net = ddpg.Actor(args).to(device)
    
    # 3. Load Weights
    # Genetic agents and RL agents use different keys in their dictionaries
    sd = torch.load(state_dict_path, map_location=device)
    
    if is_genetic_agent:
        # GeneticAgent.save_info uses 'actor_sd'
        key = 'actor_sd' 
    else:
        # DDPG.save_info uses 'actor'
        key = 'actor'

    if key in sd:
        mopderl_net.load_state_dict(sd[key])
    else:
        # Fallback: try loading directly if the file *is* the state dict
        try:
            mopderl_net.load_state_dict(sd)
        except:
            print(f"Warning: Could not load actor keys from {state_dict_path}. Available keys: {sd.keys()}")
            return None

    mopderl_net.eval()

    # 4. Inject
    wrapper_actor._impl.net = mopderl_net
    wrapper_actor._impl.max_action = 1.0
    
    return wrapper_actor

def _load_mopderl_data(
    root_dir: Path, device: torch.device
) -> Tuple[List[actors.Actor], Dict[int, Any], Dict[int, ReplayBuffer], Dict[int, np.ndarray], int]:
    
    print(f"[Stage1Loader] Scanning MOPDERL data at: {root_dir}")

    # 1. Load Args
    info_file = root_dir / "info.txt"
    args = _parse_mopderl_info_txt(info_file)
    args.device = device 

    population: List[actors.Actor] = []
    critics: Dict[int, List[torch.nn.Module]] = {}
    buffers: Dict[int, ReplayBuffer] = {}
    weights: Dict[int, np.ndarray] = {}

    # Define paths based on 'tree' output
    ckpt_dir = root_dir / "checkpoint"
    warm_up_dir = ckpt_dir / "warm_up"

    if not warm_up_dir.exists():
        raise FileNotFoundError(f"Expected warm_up directory not found at: {warm_up_dir}")

    # ====================================================
    # PART A: Load RL Agents (Teachers)
    # These provide: Actors, Critics, and Replay Buffers
    # ====================================================
    rl_agents_dir = warm_up_dir / "rl_agents"
    if rl_agents_dir.exists():
        print(f"Loading RL Agents from {rl_agents_dir}...")
        
        # Identify how many islands based on folders present (0, 1, 2...)
        island_ids = [int(p.name) for p in rl_agents_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        
        for island_id in sorted(island_ids):
            agent_path = rl_agents_dir / str(island_id)
            sd_file = agent_path / "state_dicts.pkl"
            buf_file = agent_path / "buffer.npy"

            # 1. Load Actor
            if sd_file.exists():
                actor = _load_single_actor(sd_file, args, device, pop_id=island_id, is_genetic_agent=False)
                if actor:
                    population.append(actor)

                # 2. Load Critics (Only RL Agents have these)
                sd = torch.load(sd_file, map_location=device)
                
                # Primary Critic
                mopderl_critic = ddpg.Critic(args).to(device)
                mopderl_critic.load_state_dict(sd['critic'])
                mopderl_critic.eval()

                # Secondary Critics
                sec_sd_list = sd.get('sec_critics', [])
                mopderl_sec_critics = [ddpg.Critic(args).to(device) for _ in range(len(sec_sd_list))]
                for net, net_sd in zip(mopderl_sec_critics, sec_sd_list):
                    net.load_state_dict(net_sd)
                    net.eval()
                
                # Organize: [Critic_Obj0, Critic_Obj1, ...]
                # We insert the primary critic at the index corresponding to the island_id
                all_critics = mopderl_sec_critics
                all_critics.insert(island_id, mopderl_critic)
                critics[island_id] = all_critics

            # 3. Load Buffer (Only RL Agents have large buffers worth loading)
            if buf_file.exists():
                # Mock loading to CPU first
                original_dev = args.device
                args.device = 'cpu'
                mopderl_buf = replay_memory.ReplayMemory(args.buffer_size, 'cpu')
                mopderl_buf.load_info(buf_file)
                args.device = original_dev

                # Convert to local ReplayBuffer
                catserl_buf = ReplayBuffer(
                    obs_shape=(args.state_dim,),
                    action_type="continuous",
                    action_dim=args.action_dim,
                    capacity=len(mopderl_buf.memory),
                    device=device
                )
                
                for trans in mopderl_buf.memory:
                    catserl_buf.push(
                        trans.state.squeeze(0),
                        trans.action.squeeze(0),
                        trans.reward.squeeze(0),
                        trans.next_state.squeeze(0),
                        bool(trans.done.item())
                    )
                buffers[island_id] = catserl_buf

            # 4. Infer Weights (One-Hot assumption for islands)
            w = np.zeros(getattr(args, 'num_objectives', 2))
            if island_id < len(w):
                w[island_id] = 1.0
            weights[island_id] = w

    # ====================================================
    # PART B: Load Genetic Populations (Students)
    # These provide: Actors only (pop0, pop1, ...)
    # ====================================================
    print(f"Scanning for genetic populations in {warm_up_dir}...")
    
    # Find folders named 'pop0', 'pop1', etc.
    pop_dirs = list(warm_up_dir.glob("pop*"))
    
    for p_dir in pop_dirs:
        # Extract ID from 'pop0' -> 0
        try:
            pop_id = int(p_dir.name.replace("pop", ""))
        except ValueError:
            print(f"Skipping folder {p_dir.name}, could not parse ID.")
            continue
            
        print(f"  Loading population {pop_id}...")
        
        # Iterate over individuals inside (0, 1, 2, ... 9)
        individual_dirs = [p for p in p_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        
        for ind_dir in individual_dirs:
            sd_file = ind_dir / "state_dicts.pkl"
            if sd_file.exists():
                # Note: is_genetic_agent=True handles the 'actor_sd' key difference
                actor = _load_single_actor(sd_file, args, device, pop_id=pop_id, is_genetic_agent=True)
                if actor:
                    population.append(actor)

    print(f"[Stage1Loader] Final counts: {len(population)} actors, {len(critics)} critic-sets, {len(buffers)} buffers.")
    
    # Return a default batch size (e.g. 10) as the last element
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
    critics = {int(k): [c.to(device) for c in v] for k, v in payload["critics"].items()}
    weights = {int(k): np.array(v) for k, v in payload["weights"].items()}

    # HACK
    # critics[0], critics[1] = critics[1], critics[0]
    # buffers_by_island[0], buffers_by_island[1] = buffers_by_island[1], buffers_by_island[0]

    meta = payload.get("meta", {})

    return population, critics, buffers_by_island, weights, meta