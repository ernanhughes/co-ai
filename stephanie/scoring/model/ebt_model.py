# stephanie/scoring/ebt_model.py
import torch
from torch import nn

class EBTModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)
