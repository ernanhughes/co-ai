# stephanie/scoring/ebt_model.py
import torch
from torch import nn

class DocumentEBTScorer(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.scale_factor = nn.Parameter(torch.tensor(10.0))  # Learnable scale

  
    def forward(self, ctx_emb, doc_emb):
        combined = torch.cat([ctx_emb, doc_emb], dim=-1)
        raw = self.head(combined).squeeze(-1)
        return raw * self.scale_factor