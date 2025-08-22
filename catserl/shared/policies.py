# shared/policies.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear(layer: nn.Linear) -> None:
	# Kaiming uniform for small MLPs keeps initial Q small
	nn.init.kaiming_uniform_(layer.weight, a=0.0)
	nn.init.zeros_(layer.bias)


class DiscretePolicy(nn.Module):
	def __init__(self, obs_shape: tuple[int,...], n_actions: int, hidden_dim: int = 128):
		super().__init__()
		in_dim = int(np.prod(obs_shape))
		self.net = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, n_actions)
		)
		for m in self.net:
			if isinstance(m, nn.Linear):
				_init_linear(m)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)  # [B,|A|]

	def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
		"""
		Calculates the log probability of a given action in a given state.
		"""
		# Get the logits from the policy network
		logits = self.forward(states)  # Shape: [B, N]
		
		# Use log_softmax for numerical stability
		log_probs_all = F.log_softmax(logits, dim=1)
		
		# Gather the log-probabilities of the actions that were actually taken
		log_prob = log_probs_all.gather(1, actions.long().unsqueeze(1))
		return log_prob.squeeze(1)

class ContinuousPolicy(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(ContinuousPolicy, self).__init__()

		in_dim = int(np.prod(state_dim))
		self.l1 = nn.Linear(in_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))