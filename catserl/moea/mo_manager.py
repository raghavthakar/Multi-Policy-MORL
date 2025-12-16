# mo_manager.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import uuid
import random

from scipy.spatial.distance import pdist, squareform
import torch
import numpy as np
import mo_gymnasium as mo_gym

from catserl.shared.checkpoint import Checkpoint
from catserl.shared.evo_utils.eval_pop import eval_pop
from catserl.shared.actors import Actor
from catserl.shared.buffers import MiniBuffer
from catserl.moea.basic_visualizer import BasicVisualizer
from catserl.moea.finetuners import Finetuner


__all__ = ["MOManager"]

from torch.utils.data import DataLoader, TensorDataset  # NEW: for distillation pretrain

class MOManager:
    """
    Manages the multi-objective (MO) evolutionary stage of the algorithm.

    This class is responsible for loading a population of specialist actors
    from single-objective training, and then iteratively generating new
    "child" actors to fill gaps in the Pareto front.
    """

    def __init__(
        self,
        env,
        cfg: Dict,
        ckpt_path: str | Path,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Initializes the manager by loading a merged checkpoint containing
        specialist actors, critics, and their large replay buffers.

        Args:
            env: The multi-objective environment instance.
            cfg: The main configuration dictionary for the run.
            ckpt_path: Path to the checkpoint file from Stage 1.
            device: The torch device to use for computation.
        """
        self.env = env
        self.device = torch.device(device)
        self.cfg = cfg['mopderl']
        self.num_objectives = cfg['env']['num_objectives']
        self.glob_cfg = cfg
        
        # Load all components from the end of the island stage.
        print(f"[MOManager] Loading merged checkpoint from: {ckpt_path}")
        self.ckpt = Checkpoint(ckpt_path)
        pop, critics, buffers, weights, _ = self.ckpt.load_checkpoint(device=self.device)

        # check if loaded from mopderl
        if self.ckpt.loaded_cache == False:
            # save a fast cache from mopderl save data
            self.ckpt.save_merged(pop, critics, buffers, weights, self.cfg, 2024, 4000000)

        self.population = pop
        self.critics = critics
        self.specialist_buffers = buffers

        # Store per-island scalarisation vectors (objective weights)
        self.island_weights = weights  # Dict[int, np.ndarray]

        # Sanity: assume each weight vector length == num_objectives
        example_w = next(iter(self.island_weights.values()))
        assert example_w.shape[0] == self.num_objectives, \
            f"weights dimension {example_w.shape[0]} != num_objectives {self.num_objectives}"

        # Build a mapping: objective j -> island that emphasises j the most
        self.obj_to_island: Dict[int, int] = {}
        for j in range(self.num_objectives):
            # Pick island with largest weight on objective j
            best_island = max(
                self.island_weights.keys(),
                key=lambda i: self.island_weights[i][j]
            )
            self.obj_to_island[j] = best_island

        print(f"[MOManager] Objective → island mapping: {self.obj_to_island}")


        if not self.population:
            raise ValueError("Cannot initialize MOManager: loaded population is empty.")

        self.rl_alg = self.population[0].kind
        self.finetuner = Finetuner.create('weightedmse', self.cfg['finetune'])

        print(f"[MOManager] Loaded: {len(self.population)} actors, {len(self.critics)} critics, "
              f"and {len(self.specialist_buffers)} specialist buffers.")

        self.generation = 0
        self.visualizer = BasicVisualizer(num_objectives=self.num_objectives)

        self.verify_critic_expertise()

    def _get_pareto_front(self, population: List[Actor]) -> List[Actor]:
        """
        Filters a list of actors to return only the Pareto-optimal set.
        """
        pareto_front = []
        for actor_p in population:
            is_dominated = False
            for actor_q in population:
                if actor_p is actor_q:
                    continue
                
                # An actor is dominated if another actor performs strictly better on at
                # least one objective and no worse on all other objectives.
                p_returns = actor_p.vector_return
                q_returns = actor_q.vector_return
                if np.all(q_returns >= p_returns) and np.any(q_returns > p_returns):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(actor_p)
                
        return pareto_front

    def _find_gap_and_select_parents(
        self, population: List[Actor]
    ) -> Tuple[Optional[Actor], Optional[Actor]]:
        """
        Selects a pair of parents corresponding to a Pareto-front gap.
        Instead of always picking the largest gap, we sample a gap with
        probability proportional to its Euclidean length in objective space.
        """
        evaluated_actors = [p for p in population if p.vector_return is not None]

        # Step 1: Filter the population to get only the Pareto-optimal actors.
        pareto_actors = self._get_pareto_front(evaluated_actors)
        print(f"[Parent Selection] Found {len(pareto_actors)} Pareto-optimal actors from {len(evaluated_actors)} candidates.")

        if len(pareto_actors) < 2:
            print("[Parent Selection] Not enough Pareto-optimal actors to find a gap.")
            return None, None

        # Step 2: Sort the Pareto front by the first objective to define adjacency.
        pareto_actors.sort(key=lambda p: p.vector_return[0])

        # Step 3: Compute all adjacent gaps.
        gaps = []  # list of (distance, parent_a, parent_b)
        for i in range(len(pareto_actors) - 1):
            p1 = pareto_actors[i]
            p2 = pareto_actors[i + 1]
            dist = np.linalg.norm(p1.vector_return - p2.vector_return)
            gaps.append((dist, p1, p2))

        # If somehow all distances are zero, just pick a random adjacent pair.
        total_dist = sum(d for d, _, _ in gaps)
        if total_dist <= 0.0:
            print("[Parent Selection] All gaps have zero length; selecting a random adjacent pair.")
            _, p_a, p_b = random.choice(gaps)
            return p_a, p_b

        # Step 4: Roulette-wheel selection proportional to gap length.
        r = random.random() * total_dist
        cumulative = 0.0
        for dist, p_a, p_b in gaps:
            cumulative += dist
            if r <= cumulative:
                print(f"[Parent Selection] Sampled gap of length {dist:.4f} (total length {total_dist:.4f}).")
                return p_a, p_b

        # Numerical safety fallback: return the last gap if loop didn't return.
        dist, p_a, p_b = gaps[-1]
        print(f"[Parent Selection] Fallback to last gap of length {dist:.4f}.")
        return p_a, p_b

    def _create_offspring(
        self,
        parent_a: Actor,
        parent_b: Actor,
        target_scalarisation: np.ndarray,
    ) -> Actor:
        """
        Creates a child actor and a targeted offline dataset for finetuning.
        """
        # Start the child as a clone of one parent (keeps architecture/optim state intact).
        # We'll *not* average weights; instead we'll align behavior in action space below.
        # child = parent_a.clone()
        child = Actor(
            'td3',
            pop_id=0,
            obs_shape=parent_a.obs_shape,
            action_type=parent_a.action_type,
            action_dim=parent_a.action_dim,
        )
        child.pop_id = uuid.uuid4().hex[:8]

        # Determine the number of samples to draw from each specialist buffer
        # based on the target trade-off weights.
        new_buffer_size = self.cfg['child_buffer_size']
        # Order islands according to objective index, not dict order
        island_ids = [self.obj_to_island[j] for j in range(self.num_objectives)]
        num_samples_per_buffer = (target_scalarisation * new_buffer_size).astype(int)
        
        print(f"Creating a new static dataset for child (ID: {child.pop_id}) with target size {new_buffer_size}.")

        # Sample transitions and track their origin island.
        all_s, all_a, all_r, all_s2, all_d, all_origins = [], [], [], [], [], []
        for i, island_id in enumerate(island_ids):
            num_to_sample = num_samples_per_buffer[i]
            buffer = self.specialist_buffers[island_id]

            if num_to_sample == 0 or len(buffer) == 0:
                continue
            
            num_to_sample = min(num_to_sample, len(buffer))
            print(f"  - Sampling {num_to_sample} transitions from specialist buffer {island_id}...")

            s, a, r, s2, d = buffer.sample(num_to_sample)
            all_s.append(s.cpu().numpy())
            all_a.append(a.cpu().numpy())
            all_r.append(r.cpu().numpy())
            all_s2.append(s2.cpu().numpy())
            all_d.append(d.cpu().numpy())
            
            # Store the origin island ID for each sampled transition. This is essential
            # for the finetuner to select the correct in-distribution critic.
            all_origins.append(np.full(num_to_sample, island_id))

        if not all_s:
            print("Warning: No samples were collected for the child's buffer.")
            return child

        # Assemble the final static dataset tensors.
        final_states = np.concatenate(all_s, axis=0)
        final_actions = np.concatenate(all_a, axis=0)
        final_rewards = np.concatenate(all_r, axis=0)
        final_next_states = np.concatenate(all_s2, axis=0)
        final_dones = np.concatenate(all_d, axis=0)
        final_origins = np.concatenate(all_origins, axis=0)

        # Create and populate a temporary buffer for the child.
        child_buffer = MiniBuffer(
            obs_shape=self.env.observation_space.shape,
            action_type=child.action_type,
            action_dim=child.action_dim,
            max_steps=len(final_states),
        )
        child_buffer.add_batch(
            final_states, final_actions, final_rewards, final_next_states, final_dones
        )
        child.buffer = child_buffer
        child.buffer_origins = final_origins
        
        print(f"Child's buffer created with {len(child.buffer)} total transitions.")
        
        # === Action-space distillation init ==============================
        # Briefly pretrain the child so that π_child(s) ≈ 0.5*(π_A(s)+π_B(s))
        # on *the same states* that it will be finetuned on. This avoids permutation
        # mismatch problems from raw weight-averaging and tends to land the
        # child at a reasonable behavioral midpoint before AWR finetuning.
        try:
            self._action_space_distill(child, parent_a, parent_b)
        except Exception as e:
            print(f"[Warn] Distillation pretrain skipped due to error: {e}")
        # =================================================================
        
        return child
    
    def _mutate_policy(self, child: Actor, mutation_strength: float) -> None:
        """
        Applies Gaussian noise to the parameters of a child's policy network.
        """
        with torch.no_grad():
            for param in child.policy.parameters():
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)

    def _action_space_distill(
        self,
        child: Actor,
        parent_a: Actor,
        parent_b: Actor,
    ) -> None:
        """
        Minimal, fast behavior distillation:
        Train π_child to match the average parent action on the child's buffer states.
        - Uses a few thousand gradient steps (configurable via cfg['finetune'] entries).
        - Adds tiny Gaussian noise at the end to de-correlate runs.
        """
        # Hyperparameters with safe fallbacks; keep it lightweight.
        ft_cfg = self.cfg.get('finetune', {})
        steps      = int(ft_cfg.get('pretrain_steps', 2000))
        batch_size = int(ft_cfg.get('pretrain_batch', 256))
        lr         = float(ft_cfg.get('pretrain_lr', 3e-4))
        noise_std  = float(ft_cfg.get('pretrain_noise_std', 1e-3))

        if steps <= 0 or len(child.buffer) == 0:
            return

        device = next(child.policy.parameters()).device

        # Pull all states once to avoid buffer re-sampling overhead.
        all_s, _, _, _, _ = child.buffer.sample(len(child.buffer), device=device)
        ds = TensorDataset(all_s)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        # Parents are only used for target actions; no gradients needed.
        parent_a.policy.eval()
        parent_b.policy.eval()
        child.policy.train()

        opt = torch.optim.Adam(child.policy.parameters(), lr=lr)

        # One pass = ~len(loader) updates; loop until we hit 'steps'.
        updates = 0
        while updates < steps:
            for (s_batch,) in loader:
                with torch.no_grad():
                    a_tgt = 0.5 * (parent_a.policy(s_batch) + parent_b.policy(s_batch))
                a_pred = child.policy(s_batch)
                loss = torch.mean((a_pred - a_tgt) ** 2)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(child.policy.parameters(), max_norm=1.0)
                opt.step()
                updates += 1
                if updates >= steps:
                    break

        # Tiny parameter noise to prevent identical minima across runs.
        with torch.no_grad():
            for p in child.policy.parameters():
                # Scale noise by parameter std to be architecture-agnostic.
                std = p.detach().float().std()
                if torch.isfinite(std) and std > 0:
                    p.add_(noise_std * std * torch.randn_like(p))

        child.policy.eval()
    
    def verify_critic_expertise(self, batch_size=128):
        """
        Sanity check: Verifies that critics[i][j] minimizes Bellman error 
        specifically for reward index j using data from buffer i.
        """
        print("\n--- Verifying Critic Identity (Physics Check) ---")
        
        # Iterate over every island
        for island_id, specialist_critics in self.critics.items():
            buffer = self.specialist_buffers[island_id]
            
            # Sample a diagnostic batch
            s, a, r, s2, d = buffer.sample(batch_size)
            # r is expected to be shape [batch_size, num_objectives]
            
            print(f"Island {island_id}: Checking {len(specialist_critics)} critics...")

            for claimed_obj_idx, critic in enumerate(specialist_critics):
                
                # 1. Compute target Q-value (Bootstrapping)
                # We use the critic itself for the target to keep it simple 
                # (we just want to check consistency, not train)
                with torch.no_grad():
                    # Get next action from the actor (or just current actor)
                    # Ideally use the actor from the same island, but current population actor is fine approximation
                    next_action = self.population[island_id].policy(s2) 
                    
                    if hasattr(critic, 'Q1'):
                        q_next1, q_next2 = critic(s2, next_action)
                        q_next = torch.min(q_next1, q_next2)
                        q_pred1, q_pred2 = critic(s, a)
                        q_pred = q_pred1 # Test Q1 head
                    else:
                        q_next = critic(s2, next_action).squeeze(-1)
                        q_pred = critic(s, a).squeeze(-1)

                # 2. Check Bellman Error against ALL objective rewards
                errors = []
                for test_obj_idx in range(self.num_objectives):
                    # Bellman Target using reward from test_obj_idx
                    target = r[:, test_obj_idx] + (0.99 * (1 - d)) * q_next
                    
                    # Mean Squared Bellman Error
                    msbe = ((q_pred - target) ** 2).mean().item()
                    errors.append(msbe)

                # 3. Validation
                best_fit_obj = np.argmin(errors)
                is_match = (best_fit_obj == claimed_obj_idx)
                
                status = "PASS" if is_match else "FAIL"
                print(f"  - Critic[{claimed_obj_idx}]: Best fits Obj {best_fit_obj} "
                      f"(Error: {errors[best_fit_obj]:.4f} vs Others: {np.mean(errors):.4f}) -> {status}")

                if not is_match:
                    print(f"    [WARNING] Critic at index {claimed_obj_idx} seems to predict Objective {best_fit_obj} better!")

    def evolve(self):
        """
        Executes one full generation of the multi-objective evolutionary algorithm.
        """
        print("\n--- Starting MO Generation ---")
        
        # Evaluate any policies in the population that haven't been evaluated yet.
        eval_env = mo_gym.make(self.glob_cfg['env']['name'])
        global_seed = self.glob_cfg.get("seed", 2024)
        eval_pop(
            [ind for ind in self.population if ind.vector_return is None], 
            eval_env, 
            [0.5,0.5], 
            episodes_per_actor=self.cfg['episodes_per_actor'],
            seed=global_seed + self.generation,
            max_ep_len=750,
        )

        # Print out the population's fitnesses.
        for i, ind in enumerate(self.population):
            print(f"  Actor {i} [ID: {ind.pop_id}] - Return: {ind.vector_return}, Fitness: {ind.fitness}")

        self.generation += 1
        self.visualizer.update(population=self.population, generation=self.generation)

        # log the population
        pf=self._get_pareto_front(self.population)
        self.ckpt.log_pareto_stats(self.generation, [ind.vector_return for ind in pf])

        # 1. Parent Selection: Find the largest gap in the current Pareto front.
        parent_a, parent_b = self._find_gap_and_select_parents(self.population)
        if not (parent_a and parent_b):
            print("Could not select parents. Aborting evolution step.")
            return
        print(f"Selected parents with returns: {parent_a.vector_return} <-> {parent_b.vector_return}")

        # 2. Target Calculation: Define the desired trade-off at the midpoint of the parents.
        evaluated = [p for p in self.population if p.vector_return is not None]
        returns = np.vstack([p.vector_return for p in evaluated])
        mins, maxs = returns.min(axis=0), returns.max(axis=0)
        
        denom = np.where(maxs - mins == 0.0, 1.0, maxs - mins)
        norm_a = (parent_a.vector_return - mins) / denom
        norm_b = (parent_b.vector_return - mins) / denom
        
        midpoint = 0.5 * (norm_a + norm_b)
        target_scalarisation = midpoint / np.sum(midpoint) if np.sum(midpoint) > 1e-8 else np.ones(self.num_objectives) / self.num_objectives
        print(f"Calculated target weights: {target_scalarisation}")

        # 3. Offspring Creation: Generate a new child and its training data.
        child = self._create_offspring(parent_a, parent_b, target_scalarisation)
        
        # 4. Fine-tuning: Use the offline learning process to train the child.
        print("--- Fine-tuning Child Actor ---")
        self.finetuner.offline_finetune(
            child=child,
            target_scalarisation=target_scalarisation,
            critics=self.critics
        )
        
        # Add the new, fine-tuned child to the population for the next generation.
        self.population.append(child)
