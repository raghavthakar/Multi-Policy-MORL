# catserl/shared/rl.py
from __future__ import annotations
"""Minimal pluggable‑algorithm RL worker

This module keeps **all** algorithm logic in one file while allowing you to
swap RL algorithms by passing a string (e.g. "dqn", "ppo") to ``RLWorker``.
"""

from typing import Tuple, Dict, List
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from catserl.shared.policies import ContinuousPolicy
from catserl.shared.buffers import ReplayBuffer 

# -----------------------------------------------------------------------------
# Tiny *informal* interface – every algorithm must expose these five methods.
# -----------------------------------------------------------------------------

class Algo:  
	def act(self, state: np.ndarray) -> int: 
		raise NotImplementedError

	def remember(self, *transition) -> None: 
		raise NotImplementedError

	def update(self) -> None: 
		raise NotImplementedError

	def save_state(self) -> Dict:
		raise NotImplementedError

	def load_state(self, state_dict: Dict) -> None:
		raise NotImplementedError

# -----------------------------------------------------------------------------
# TD3 Implementation
# -----------------------------------------------------------------------------

class TD3Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(TD3Critic, self).__init__()

		# Q1 architecture
		# MODIFIED: Removed weight conditioning logic. 
		# state_dim is now just the observation dimension.
		in_dim = int(np.prod(state_dim))
		
		self.l1 = nn.Linear(in_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(in_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

	def forward(self, state, action):
		# MODIFIED: No concatenation of weights here.
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1

class TD3(Algo):
	def __init__(
		self,
		obs_shape: Tuple[int, ...],
		action_type: str,
		action_dim: int,
		max_action: float,
		num_secondary_critics: int,
		cfg: dict,
		device: torch.device,
	) -> None:
		"""
		Args:
			num_secondary_critics: int
				The number of secondary objectives (N-1) this agent needs to track.
		"""
		self.device = device
		self.n_actions = action_dim
		
		# Load parameters from cfg
		td3_cfg = cfg['td3']
		self.discount = float(td3_cfg.get("discount", 0.99))
		self.tau = float(td3_cfg.get("tau", 0.005))
		self.policy_noise = float(td3_cfg.get("policy_noise", 0.2))
		self.noise_clip = float(td3_cfg.get("noise_clip", 0.5))
		self.policy_freq = int(td3_cfg.get("policy_freq", 2))
		self.exploration_noise = float(td3_cfg.get("exploration_noise", 0.1))
		self.rl_kick_in_frames = int(td3_cfg.get("start_timesteps", 25000))
		self.batch_size = int(td3_cfg.get("batch_size", 256))
		self.buffer_size = int(td3_cfg.get("buffer_size", 1_000_000))
		critic_lr = float(td3_cfg.get("critic_lr", 3e-4))
		actor_lr = float(td3_cfg.get("actor_lr", 8e-5))

		self.max_action = max_action

		# --- Actor ---
		self.actor = ContinuousPolicy(obs_shape, self.n_actions, max_action).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)

		# --- Primary Critic (Optimizes for Main Objective) ---
		# MODIFIED: Input is just obs_shape, no weight conditioning
		self.critic = TD3Critic(obs_shape, self.n_actions).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

		# --- Secondary Critics (Evaluate N-1 Other Objectives) ---
		# MODIFIED: List of independent critics for other objectives
		self.sec_critics = nn.ModuleList([
			TD3Critic(obs_shape, self.n_actions).to(self.device) 
			for _ in range(num_secondary_critics)
		])
		self.sec_critics_target = copy.deepcopy(self.sec_critics)
		self.sec_critics_optimizer = [
			torch.optim.Adam(sc.parameters(), critic_lr) for sc in self.sec_critics
		]

		# Replay buffer
		self.buffer = ReplayBuffer(
			obs_shape,
			action_type,
			action_dim,
			capacity=int(cfg.get("buffer_size", 1_000_000)),
			device=device,
		)

		self.total_it = 0
		self.frames_idx = 0

	def act(self, state: np.ndarray, noisy_action=True, random_action=False):
		self.frames_idx += 1

		if random_action:
			return np.random.uniform(-self.max_action, self.max_action, self.n_actions)
		
		state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state_tensor).cpu().data.numpy().flatten()
		
		if not noisy_action:
			return action.clip(-self.max_action, self.max_action)

		noise = np.random.normal(0, self.max_action * self.exploration_noise, size=self.n_actions)
		action_with_noise = action + noise
		return action_with_noise.clip(-self.max_action, self.max_action)
	
	def remember(self, *transition) -> None:
		self.buffer.push(*transition)

	def _update_single_critic(
		self, 
		critic_net: nn.Module, 
		target_net: nn.Module, 
		optimizer: torch.optim.Optimizer, 
		state: torch.Tensor, 
		action: torch.Tensor, 
		next_state: torch.Tensor, 
		scalar_reward: torch.Tensor, 
		done_2d: torch.Tensor
	):
		"""Helper to perform standard TD3 critic update on a specific network."""
		with torch.no_grad():
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Target Q-values
			target_Q1, target_Q2 = target_net(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = scalar_reward + (1 - done_2d) * self.discount * target_Q

		# Current Q-values
		current_Q1, current_Q2 = critic_net(state, action)

		# Loss and Optimize
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()

	def update(self, main_scalar_weight: np.ndarray, other_scalar_weights: list[np.ndarray]) -> None:
		self.total_it += 1
		state, action, r_vec, next_state, done = self.buffer.sample(self.batch_size)
		action = action.view(self.batch_size, -1)
		done_2d = done.unsqueeze(1)

		# 1. Update Primary Critic (Conditioned on Main Weight)
		main_scalar_weight_t = torch.from_numpy(main_scalar_weight).float().to(self.device)
		# Reward shape: (Batch, 1)
		main_reward = (r_vec * main_scalar_weight_t).sum(dim=1, keepdim=True)

		self._update_single_critic(
			self.critic, self.critic_target, self.critic_optimizer,
			state, action, next_state, main_reward, done_2d
		)

		# 2. Update Secondary Critics (Conditioned on N-1 Other Weights)
		# We assume self.sec_critics aligns by index with other_scalar_weights
		for i, other_w in enumerate(other_scalar_weights):
			if i >= len(self.sec_critics): break # Safety check
			
			other_w_t = torch.from_numpy(other_w).float().to(self.device)
			other_reward = (r_vec * other_w_t).sum(dim=1, keepdim=True)
			
			self._update_single_critic(
				self.sec_critics[i], self.sec_critics_target[i], self.sec_critics_optimizer[i],
				state, action, next_state, other_reward, done_2d
			)

		# 3. Delayed Actor Update
		# MODIFIED: Actor is updated ONLY using the Primary Critic
		if self.total_it % self.policy_freq == 0:
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Soft update Primary Networks
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# Soft update Secondary Networks
			for i in range(len(self.sec_critics)):
				for param, target_param in zip(self.sec_critics[i].parameters(), self.sec_critics_target[i].parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# Soft update Actor
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def get_critic(self):
		return self.critic_target
	
	def get_buffer(self):
		return self.buffer

	def save_state(self) -> Dict:
		# MODIFIED: Save secondary critics
		return {
			'actor': self.actor.state_dict(),
			'critic': self.critic.state_dict(),
			'sec_critics': [c.state_dict() for c in self.sec_critics],
			'actor_optimizer': self.actor_optimizer.state_dict(),
			'critic_optimizer': self.critic_optimizer.state_dict(),
			'sec_critics_optimizers': [opt.state_dict() for opt in self.sec_critics_optimizer],
			'total_it': self.total_it,
			'frames_idx': self.frames_idx
		}

	def load_state(self, state_dict: Dict) -> None:
		self.actor.load_state_dict(state_dict['actor'])
		self.actor_target = copy.deepcopy(self.actor)
		
		# Primary Load
		self.critic.load_state_dict(state_dict['critic'])
		self.critic_target = copy.deepcopy(self.critic)
		
		# Secondary Load
		if 'sec_critics' in state_dict:
			for i, sd in enumerate(state_dict['sec_critics']):
				if i < len(self.sec_critics):
					self.sec_critics[i].load_state_dict(sd)
					# Re-copy to target to ensure sync
					self.sec_critics_target[i] = copy.deepcopy(self.sec_critics[i])

		self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
		self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
		
		if 'sec_critics_optimizers' in state_dict:
			for i, sd in enumerate(state_dict['sec_critics_optimizers']):
				if i < len(self.sec_critics_optimizer):
					self.sec_critics_optimizer[i].load_state_dict(sd)

		self.total_it = state_dict['total_it']
		self.frames_idx = state_dict['frames_idx']
	
	def export_policy_params(self):
		"""
		Exports the actor's policy parameters by calling the method
		on the ContinuousPolicy instance.
		"""
		return self.actor.export_params()
	
	def get_primary_critic(self):
		return self.critic

	def get_secondary_critics(self):
		return self.sec_critics

# -----------------------------------------------------------------------------
# RLWorker
# -----------------------------------------------------------------------------

class RLWorker:
	def __init__(
		self,
		algo: str,
		obs_shape: Tuple[int, ...],
		action_type: str,
		action_dim: int,
		max_action: float,
		scalar_weight: np.ndarray,
		other_scalar_weights: list[np.ndarray],
		cfg: dict,
		device: torch.device,
	) -> None:
		self.main_scalar_weight = scalar_weight 
		self.other_scalar_weights = other_scalar_weights 
		
		algo = algo.lower()
		if algo == "td3":
			# MODIFIED: Pass number of secondary critics to init
			self.agent = TD3(
				obs_shape, 
				action_type, 
				action_dim, 
				max_action, 
				num_secondary_critics=len(other_scalar_weights),
				cfg=cfg, 
				device=device
			)
		else:
			raise ValueError(f"Unknown algorithm '{algo}'")
		
		self.device = device

	def act(self, state: np.ndarray, noisy_action=True, random_action=False) -> int:
		return self.agent.act(state, noisy_action=noisy_action, random_action=random_action)

	def remember(self, *transition) -> None:
		self.agent.remember(*transition)

	def update(self) -> None:
		self.agent.update(self.main_scalar_weight, self.other_scalar_weights)

	def save_state(self) -> Dict:
		return self.agent.save_state()

	def load_state(self, state_dict: Dict) -> None:
		self.agent.load_state(state_dict)

	def export_policy_params(self):
		return getattr(self.agent, "export_policy_params", None)()
	
	def critic(self):
		return self.agent.get_critic()
	
	def critics(self) -> List[nn.Module]:
		"""
		Returns a list of all critics sorted by their objective index.
		
		If Main=[0,1] (Obj 2) and Other=[1,0] (Obj 1):
		Returns: [Secondary_Critic(Obj 1), Primary_Critic(Obj 2)]
		"""
		pairs = []

		# 1. Add Primary Critic and its weight
		# active_agent.critic is the primary network
		pairs.append((self.main_scalar_weight, self.agent.get_primary_critic()))

		# 2. Add Secondary Critics and their weights
		# We zip the stored weights with the actual networks
		sec_critics = self.agent.get_secondary_critics()
		for weight, critic in zip(self.other_scalar_weights, sec_critics):
			pairs.append((weight, critic))

		# 3. Sort based on the active index of the one-hot vector
		# np.argmax([0, 1]) -> 1
		# np.argmax([1, 0]) -> 0
		pairs.sort(key=lambda x: np.argmax(x[0]))

		# 4. Return just the ordered critics
		return [c for w, c in pairs]
	
	def buffer(self):
		return self.agent.get_buffer()