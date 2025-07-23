# stephanie/scoring/model/ebt_model.py
import torch
from torch import nn


class EBTModel(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim: int = 256, num_actions: int = 5):
        """
        Energy-Based Transformer Model with optional policy head.
        Args:
            embedding_dim: Input dimension of embeddings.
            hidden_dim: Hidden layer dimension.
            num_actions: Number of action logits to output for policy inference.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # A small feedforward head that maps concatenated (goal + doc) embeddings to a single score
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),  # Input: goal + doc embeddings
            nn.ReLU(),
            nn.Linear(256, 1),  # Output: scalar score (before scaling)
        )
        # Learnable scaling factor to adjust output magnitude during training
        self.scale_factor = nn.Parameter(torch.tensor(10.0))

    def forward(self, context_emb, output_emb):
        # your encoding logic here
        combined = context_emb * output_emb  # or whatever you use
        energy = self.energy_head(combined)

        # New policy logits (for GILD)
        logits = self.policy_head(combined)

        return {
            "energy": energy.squeeze(-1),       # ensure 1D
            "action_logits": logits              # shape [batch, num_actions]
        }
