# shared/policies.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal


def _init_linear(layer: nn.Linear) -> None:
	# Kaiming uniform for small MLPs keeps initial Q small
	nn.init.kaiming_uniform_(layer.weight, a=0.0)
	nn.init.zeros_(layer.bias)

class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ContinuousPolicy, self).__init__()

        in_dim = int(np.prod(state_dim))
        self.l1 = nn.Linear(in_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        
        log_std_init = -0.5 * np.ones(action_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std_init), requires_grad=False)


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_log_prob(self, state, action):
        mean = self.forward(state)
        std = torch.exp(self.log_std)
        base_dist = Normal(mean, std)
        dist = Independent(base_dist, 1)
        return dist.log_prob(action)
        
    def export_params(self) -> tuple[torch.Tensor, int]:
        """
        Flattens and returns all network parameters using a generic, robust method.
        This is the symmetrical counterpart to the Actor's load_flat_params.
        """
        with torch.no_grad():
            # Use the generic .parameters() iterator to guarantee symmetry.
            # This automatically includes all layers and parameters like self.log_std.
            flat_params = torch.cat([
                p.data.view(-1) for p in self.parameters()
            ]).cpu().clone()

        # The hidden dimension can be retrieved directly from the first layer.
        hidden_dim = self.l1.out_features
        return flat_params, hidden_dim