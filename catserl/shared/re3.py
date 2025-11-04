from typing import Tuple

import torch
import torch.nn as nn
import numpy as np

class RE3:
    def __init__(self, obs_shape, embed_dim=128, k=3):
        self.beta = 0.05             # Scaling factor for intrinsic reward
        self.beta_decay = 0.00001     # Linear decay rate per update
        self.batch_size = 512
        self.k = k                   # Number of nearest neighbors
        
        obs_dim = np.prod(obs_shape)  # Flattened input dimension if observation is multi-dimensional

        # Frozen random encoder for state embeddings
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

    def compute_intrinsic(self, current_state: torch.Tensor, sample: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Compute the intrinsic reward for a single state using kNN entropy estimate.

        Parameters
        ----------
        current_state : torch.Tensor
            A tensor of shape [obs_dim], the current state for which to compute the intrinsic reward.
        sample : Tuple
            A tuple (s, a, r, s2, d) from the replay buffer; only s (states) is used.

        Returns
        -------
        intrinsic_reward : torch.Tensor
            A scalar tensor representing the intrinsic reward for the given state.
        """
        s, a, r, s2, d = sample

        with torch.no_grad():
            # Encode the current state and the batch
            current_embedding = self.encoder(current_state.unsqueeze(0))  # [1, D]
            batch_embeddings = self.encoder(s)  # [B, D]

            # Normalize for cosine-stable L2 distances
            # current_embedding = torch.nn.functional.normalize(current_embedding, p=2, dim=1)
            # batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

            # Compute distances: [B]
            dists = torch.cdist(current_embedding, batch_embeddings, p=2).squeeze(0)  # shape: [B]

            # Sort and select top-k smallest distances
            knn_dists, _ = torch.sort(dists)
            k = min(self.k, len(knn_dists))
            top_k_dists = knn_dists[:k]

            # Intrinsic reward: average log-distance (RE3-style)
            log_dists = torch.log(top_k_dists + 1.0)
            entropy_bonus = log_dists.mean()

            # Scale and decay
            intrinsic_reward = self.beta * entropy_bonus
            self.beta *= (1.0 - self.beta_decay)

        return intrinsic_reward
