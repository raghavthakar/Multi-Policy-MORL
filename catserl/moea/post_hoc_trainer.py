import torch
import numpy as np
import copy
import mo_gymnasium as mo_gym

from catserl.shared.evo_utils.eval_pop import eval_pop

class PostHocTrainer:
    def __init__(self, cfg):
        self.glob_cfg = cfg

    def train_post_hoc_critics(self, critics, buffers, actors):
        print("\n--- Starting Post-Hoc Critic Training (Cross-Evaluation) ---")
        
        # 1. Extract Primary Critics & Determine Objective Count
        primary_critics = {}
        for obj_idx in critics.keys():
            # Assume structure is critics[island_id][obj_id]
            # The primary critic is where island_id == obj_id
            primary_critics[obj_idx] = critics[obj_idx][obj_idx]
        
        num_objectives = len(primary_critics)
        device = next(primary_critics[0].parameters()).device

        # 2. Evaluate Population to find the "Specialist" (Owner) for each objective
        # We need to know: Which actor generated Buffer 0? Which generated Buffer 1?
        print("  > Identifying specialist policies...")
        eval_env = mo_gym.make(self.glob_cfg['env']['name'])
        global_seed = self.glob_cfg.get("seed", 2024)
        
        # Only evaluate actors that haven't been evaluated yet
        unevaluated = [a for a in actors if a.vector_return is None]
        if unevaluated:
            eval_pop(
                unevaluated, 
                eval_env, 
                np.ones(num_objectives)/num_objectives, 
                episodes_per_actor=self.glob_cfg['mopderl']['episodes_per_actor'],
                seed=global_seed,
                max_ep_len=750,
            )

        # Map each objective/island to the best actor found for that objective.
        # This actor will serve as the fixed 'target policy' for bellman updates.
        specialist_policies = {}
        vector_returns = np.array([policy.vector_return for policy in actors])
        
        for obj_idx in range(num_objectives):
            # Fix: Create correct one-hot vector
            w = np.zeros(num_objectives)
            w[obj_idx] = 1.0
            
            # Scalarize and find max
            scalar_returns = np.sum(vector_returns * w, axis=1)
            specialist_idx = np.argmax(scalar_returns)
            specialist_policies[obj_idx] = actors[specialist_idx]
            
            print(f"    - Specialist for Obj {obj_idx}: Actor {actors[specialist_idx].pop_id} (Return: {actors[specialist_idx].vector_return})")

        # 3. The Main Cross-Training Loop
        # We need to fill the matrix: critics[island_id][target_obj]

        import torch.nn.functional as F

        # Hyperparameters for this supervised regression phase
        train_steps = 5000   # Sufficient for policy evaluation on fixed buffer
        batch_size = 256
        lr = 3e-4
        gamma = 0.99
        tau = 0.005

        for island_id in range(num_objectives):
            source_buffer = buffers[island_id]
            target_actor = specialist_policies[island_id] # Fixed policy pi_i

            # Ensure the list for this island has enough slots
            if len(critics[island_id]) < num_objectives:
                # Extend list with placeholders if needed
                critics[island_id].extend([None] * (num_objectives - len(critics[island_id])))

            for target_obj in range(num_objectives):
                # If we are looking at the island's own objective, we already have the critic.
                if island_id == target_obj:
                    continue

                print(f"  > Training Secondary Critic: Island {island_id} (Source) evaluating on Objective {target_obj} (Target)...")

                # A. Instantiate Network
                # Clone the architecture of the primary critic, but reset weights
                primary = primary_critics[island_id]
                new_critic = copy.deepcopy(primary)
                new_critic.apply(self._reset_weights) # Helper to re-init weights
                new_critic.to(device)
                
                target_critic = copy.deepcopy(new_critic)
                target_critic.load_state_dict(new_critic.state_dict())
                
                optimizer = torch.optim.Adam(new_critic.parameters(), lr=lr)
                
                # B. Fitted Q-Iteration Loop
                for step in range(train_steps):
                    # Sample batch
                    s, a, r_vec, s2, d = source_buffer.sample(batch_size)
                    
                    # --- FIX: Ensure shapes are strictly [B, 1] ---
                    # Extract single scalar reward for the target objective
                    r = r_vec[:, target_obj].unsqueeze(1) # [256, 1]
                    
                    # Force 'd' to be [256, 1] to prevent broadcasting explosion
                    d = d.view(-1, 1) 
                    # ----------------------------------------------

                    with torch.no_grad():
                        # Get next action from the fixed specialist policy
                        next_action = target_actor.policy(s2)
                        
                        # Compute Target Q
                        if hasattr(target_critic, 'Q1'):
                            tq1, tq2 = target_critic(s2, next_action)
                            target_q = torch.min(tq1, tq2)
                        else:
                            target_q = target_critic(s2, next_action)
                        
                        # Ensure target_q is [256, 1] (sometimes squeezing happens in forward)
                        target_q = target_q.view(-1, 1)

                        # Standard Bellman target
                        # Now: [256, 1] + (scalar * [256, 1] * [256, 1]) -> [256, 1]
                        y = r + (gamma * (1 - d) * target_q)

                    # Compute Current Q and Loss
                    if hasattr(new_critic, 'Q1'):
                        q1, q2 = new_critic(s, a)
                        # Ensure predictions are also [256, 1]
                        q1, q2 = q1.view(-1, 1), q2.view(-1, 1) 
                        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                    else:
                        q = new_critic(s, a)
                        q = q.view(-1, 1)
                        loss = F.mse_loss(q, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    print(step, loss.item())

                    # Soft update target network
                    if step % 2 == 0:
                        for p, tp in zip(new_critic.parameters(), target_critic.parameters()):
                            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

                # C. Save the trained critic
                critics[island_id][target_obj] = new_critic
                print(f"    - Finished training. Final Loss: {loss.item():.5f}")

        print("--- Post-Hoc Critic Training Complete ---\n")
        return critics

    @staticmethod
    def _reset_weights(m):
        import torch.nn as nn
        """Resets weights to random initialization for new critics."""
        if isinstance(m, nn.Linear):
            # Use PyTorch default (Kaiming Uniform) or orthogonal
            nn.init.orthogonal_(m.weight, gain=1.414)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)