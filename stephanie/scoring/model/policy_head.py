# stephanie/scoring/model/policy_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU


class PolicyHead(nn.Module):
    def __init__(self, zsa_dim, hdim, num_actions=3):
        """
        Policy head with advantage-weighted regression
        
        Args:
            zsa_dim: Dimension of encoded state-action vector
            hdim: Hidden layer dimension
            num_actions: Number of discrete actions (ebt, svm, mrq)
        """
        super().__init__()
        self.linear = nn.Sequential(
            Linear(zsa_dim, hdim),
            ReLU(),
            Linear(hdim, num_actions)
        )
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, zsa):
        """
        Predict policy logits for action selection
        
        Args:
            zsa: Encoded state-action vector
        Returns:
            Policy logits (unnormalized action probabilities)
        """
        return self.linear(zsa)