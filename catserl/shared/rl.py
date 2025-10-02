# catserl/shared/rl.py
from __future__ import annotations
"""Minimal pluggable‑algorithm RL worker

This module keeps **all** algorithm logic in one file while allowing you to
swap RL algorithms by passing a string (e.g. "dqn", "ppo") to ``RLWorker``.

The only abstraction is a tiny *duck‑typed* ``Algo`` base class that defines
five methods.  Each concrete algorithm class (``DQN`` below) implements those
methods so the worker can delegate to it.

Add new algorithms by:
1. Creating another class that follows the same interface.
2. Adding an ``elif`` branch in ``RLWorker.__init__``.

No global registry or separate packages are required.
"""

from typing import Tuple, Dict
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from catserl.shared.policies import ContinuousPolicy

# -----------------------------------------------------------------------------
# Tiny *informal* interface – every algorithm must expose these five methods.
# -----------------------------------------------------------------------------


class Algo:  # noqa: D101 – doctring would just repeat the method names
	"""Lightweight base class documenting the required public surface.

	We do **not** use ABC / abstractmethod to avoid an extra import and keep
	the dependency graph flat – algorithms simply override the methods they
	need.  If a method is left unimplemented we want Python to raise a
	``NotImplementedError`` at runtime, which mirrors an ABC anyway.
	"""

	# Each concrete class should provide explicit type hints to satisfy mypy.
	def act(self, state: np.ndarray) -> int:  # noqa: D401 – imperative mood
		raise NotImplementedError

	def remember(self, *transition) -> None:  # noqa: D401 – imperative mood
		raise NotImplementedError

	def update(self) -> None:  # noqa: D401 – imperative mood
		raise NotImplementedError

	def save_state(self) -> Dict:
		raise NotImplementedError

	def load_state(self, state_dict: Dict) -> None:
		raise NotImplementedError

from catserl.shared.buffers import ReplayBuffer  # isort: skip  (project import)

class TD3Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(TD3Critic, self).__init__()

		# Q1 architecture
		in_dim = int(np.prod(state_dim))
		self.l1 = nn.Linear(in_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(in_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
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

class TD3(Algo): # Inherit from Algo
	def __init__(
		self,
		obs_shape: Tuple[int, ...],
		action_type: str,
		action_dim: int,
		max_action: float,
		scalar_weight_like: np.ndarray,
		cfg: dict,
		device: torch.device,
	) -> None:
		"""
		Args:
			scalar_weight_like: np.ndarray
				An example scalar weight vector to determine the shape.
				We do not use the actual scalar weight here because
				it may be different for each island.
		"""
		self.device = device
		self.n_actions = action_dim # Note: n_actions is a bit of a misnomer here
		
		# Load parameters from cfg for consistency and correctness
		td3_cfg = cfg['td3'] # Get the 'td3' sub-dictionary
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

		# Networks ----------------------------------------------------
		self.actor = ContinuousPolicy(obs_shape, self.n_actions, max_action).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)

		# Scalar weight conditioned critic
		# Calculate the flattened dimension for the state and the weight vector.
		flat_obs_dim = int(np.prod(obs_shape))
		weight_dim = scalar_weight_like.shape[0]
		critic_state_dim = flat_obs_dim + weight_dim
		
		# Initalise the critic and its target network and optimiser
		self.critic = TD3Critic((critic_state_dim,), self.n_actions).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

		self.max_action = max_action

		# Replay buffer ----------------------------------------------
		self.buffer = ReplayBuffer(
			obs_shape,
			action_type,
			action_dim,
			capacity=int(cfg.get("buffer_size", 1_000_000)),
			device=device,
		)

		self.total_it = 0 # How many updates have been made
		self.frames_idx = 0 # How many frames have been collected

	def act(self, state: np.ndarray, noisy_action=True, random_action=False):
		self.frames_idx += 1

		# If RL has not kicked in then take a random action
		if random_action:
			return np.random.uniform(-self.max_action, self.max_action, self.n_actions)
		
		# Step 1: Get the deterministic action from the policy
		state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state_tensor).cpu().data.numpy().flatten()
		
		# Deterministic action for policy evaluation.
		if not noisy_action:
			return action.clip(-self.max_action, self.max_action)

		# Step 2: Add scaled Gaussian noise for exploration
		noise = np.random.normal(
			0, 
			self.max_action * self.exploration_noise, 
			size=self.n_actions
		)
		action_with_noise = action + noise
	
		# Step 3: Clip the final action to ensure it's within the valid range
		clipped_action = action_with_noise.clip(-self.max_action, self.max_action)

		return clipped_action
	
	def remember(self, *transition) -> None:
		self.buffer.push(*transition)

	def update(self, main_scalar_weight: np.ndarray, other_scalar_weights: list[np.ndarray]) -> None:
		"""
		Performs a single training step for the TD3 agent.

		This method updates the critic and actor networks. The critic is conditioned
		on a scalarization weight and is trained to predict Q-values for all
		provided objective weightings. The actor is updated less frequently and
		is optimized only with respect to its island's main objective.
		"""
		self.total_it += 1
		state, action, r_vec, next_state, done = self.buffer.sample(self.batch_size)
		action = action.view(self.batch_size, -1)

		# --- Critic Update ---
		# The critic is trained to predict Q-values for multiple objectives, conditioned
		# on the scalarization weight. We loop through each weight vector provided
		# and compute a separate loss to make the critic "multi-objective aware".
		all_scalar_weights = [main_scalar_weight] + other_scalar_weights
		for scalar_weight_np in all_scalar_weights:
			# Prepare the scalarization weight tensor for batch operations. This involves
			# converting it from NumPy, moving to the correct device, and expanding
			# its dimensions to match the batch size of the state tensors.
			scalar_weight = torch.from_numpy(scalar_weight_np).float().to(self.device)
			scalar_weight_batch = scalar_weight.expand(self.batch_size, -1)

			# Create the conditioned input for the critic by concatenating the
			# state/next_state and the corresponding weight vector.
			state_conditioned = torch.cat([state, scalar_weight_batch], 1)
			next_state_conditioned = torch.cat([next_state, scalar_weight_batch], 1)

			# Compute the scalar reward for the current objective weighting.
			reward = (r_vec * scalar_weight).sum(dim=1, keepdim=True)
			done_2d = done.unsqueeze(1)

			with torch.no_grad():
				# Target policy smoothing: add clipped noise to the target actor's actions
				# to prevent function approximation errors from propagating.
				noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
				next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

				# Compute the target Q-value using the clipped double-Q trick to
				# mitigate overestimation bias.
				target_Q1, target_Q2 = self.critic_target(next_state_conditioned, next_action)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + (1 - done_2d) * self.discount * target_Q

			# Get the current Q-value estimates.
			current_Q1, current_Q2 = self.critic(state_conditioned, action)

			# Compute and optimize the critic loss based on the TD error.
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

		# --- Delayed Actor and Target Network Updates ---
		if self.total_it % self.policy_freq == 0:
			# The actor is only optimized with respect to the island's main objective.
			# We prepare a conditioned state using only the main scalarization weight.
			main_scalar_weight_tensor = torch.from_numpy(main_scalar_weight).float().to(self.device)
			main_scalar_weight_batch = main_scalar_weight_tensor.expand(self.batch_size, -1)
			state_conditioned_for_actor = torch.cat([state, main_scalar_weight_batch], 1)
			
			# Compute the actor loss, which aims to maximize the critic's Q-value estimate.
			actor_loss = -self.critic.Q1(state_conditioned_for_actor, self.actor(state)).mean()
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Soft update the target networks using Polyak averaging.
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				
	def export_policy_params(self):
		"""
		Exports the actor's policy parameters by calling the method
		on the ContinuousPolicy instance.
		"""
		return self.actor.export_params()
	
	def get_critic(self):
		return self.critic_target
	
	def get_buffer(self):
		return self.buffer

	# MODIFIED: Add save/load methods to satisfy the Algo interface
	def save_state(self) -> Dict:
		"""
		Exports the complete state of the agent to a dictionary.

		This includes network weights, optimizer states, and training progress,
		which is essential for correctly resuming training.
		"""
		return {
			'actor': self.actor.state_dict(),
			'critic': self.critic.state_dict(),
			'actor_optimizer': self.actor_optimizer.state_dict(),
			'critic_optimizer': self.critic_optimizer.state_dict(),
			'total_it': self.total_it,
			'frames_idx': self.frames_idx
		}

	def load_state(self, state_dict: Dict) -> None:
		"""
		Restores the agent's state from a dictionary.
		"""
		self.actor.load_state_dict(state_dict['actor'])
		self.actor_target = copy.deepcopy(self.actor)
		self.critic.load_state_dict(state_dict['critic'])
		self.critic_target = copy.deepcopy(self.critic)
		self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
		self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
		self.total_it = state_dict['total_it']
		self.frames_idx = state_dict['frames_idx']

# -----------------------------------------------------------------------------
# RLWorker – orchestration shell that delegates to a chosen algorithm class
# -----------------------------------------------------------------------------

class RLWorker:
	"""Environment‑agnostic worker that can run multiple RL algorithms.

	The worker owns a **single** concrete algorithm instance.  Delegation keeps
	public signatures identical to the original design so existing training
	loops require zero changes beyond specifying the ``algo`` string.
	"""

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
		self.main_scalar_weight = scalar_weight # the primary scalarised objective that the island is optimising for
		self.other_scalar_weights = other_scalar_weights # the other scalarisations that the island is not optimising for but we still want to update the critic for
		
		algo = algo.lower()
		if algo == "td3":
			self.agent = TD3(obs_shape, action_type, action_dim, max_action, scalar_weight, cfg, device)
		else:
			raise ValueError(
				f"Unknown algorithm '{algo}'. Add an elif branch in RLWorker.__init__."
			)
		
		self.device = device

	# ------------------------------------------------------------------
	# Thin delegation layer – forwards calls to the underlying algorithm.
	# ------------------------------------------------------------------
	def act(self, state: np.ndarray, noisy_action=True, random_action=False) -> int:  # noqa: D401 – imperative mood
		return self.agent.act(state, noisy_action=noisy_action, random_action=random_action)

	def remember(self, *transition) -> None:  # noqa: D401 – imperative mood
		self.agent.remember(*transition)

	def update(self) -> None:  # noqa: D401 – imperative mood
		self.agent.update(self.main_scalar_weight, self.other_scalar_weights)

	def save_state(self) -> Dict:
		"""Pass-through method to get the agent's state dictionary."""
		return self.agent.save_state()

	def load_state(self, state_dict: Dict) -> None:
		"""Pass-through method to load the agent's state from a dictionary."""
		self.agent.load_state(state_dict)

	def export_policy_params(self):  # noqa: D401 – imperative mood
		return getattr(self.agent, "export_policy_params", None)()
	
	def critic(self):
		return self.agent.get_critic()
	
	def buffer(self):
		return self.agent.get_buffer()
