# stephanie/embeddings/hnet_embedder.py
import torch.nn as nn
import torch
import numpy as np
from stephanie.protocols.embeddings import EmbeddingProtocol


class ByteLevelTokenizer:
    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))  # Raw byte tokenization

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")
    import torch

class ChunkBoundaryPredictor(nn.Module):
    def __init__(self, vocab_size=256, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True)
        self.boundary_scorer = nn.Linear(hidden_dim * 2, 1)

    def forward(self, tokens: list[int]) -> torch.Tensor:
        x = torch.tensor(tokens).long()
        x = self.embedding(x)
        x, _ = self.lstm(x.unsqueeze(1))
        scores = self.boundary_scorer(x)
        return scores.sigmoid().flatten()
    
class StephanieHNetChunker:
     def __init__(self, boundary_predictor=None, threshold=0.7):
        self.tokenizer = ByteLevelTokenizer()
        self.boundary_predictor = boundary_predictor or ChunkBoundaryPredictor()
        self.threshold = threshold

     def chunk(self, text: str) -> list:
        tokens = self.tokenizer.tokenize(text)
        with torch.no_grad():
            scores = self.boundary_predictor(torch.tensor(tokens).unsqueeze(0).long())
        boundaries = (scores > self.threshold).nonzero(as_tuple=True)[0].tolist()
        chunks = []
        prev = 0
        for b in boundaries:
            chunk_tokens = tokens[prev:b+1]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            prev = b + 1

        # Add final chunk
        if prev < len(tokens):
            final_chunk = self.tokenizer.decode(tokens[prev:])
            chunks.append(final_chunk)

        return chunks
    
class PoolingStrategy:
    @staticmethod
    def mean_pool(embeddings: list[list[float]]) -> list[float]:
        return np.mean(embeddings, axis=0).tolist()

    @staticmethod
    def weighted_mean_pool(embeddings: list[list[float]], weights: list[float]) -> list[float]:
        return np.average(embeddings, weights=weights, axis=0).tolist()
    


class StephanieHNetEmbedder(EmbeddingProtocol):
    def __init__(self, embedder: EmbeddingProtocol):
        self.chunker = StephanieHNetChunker()
        self.embedder = embedder
        self.pooler = PoolingStrategy()

    def embed(self, text: str) -> list[float]:
        chunks = self.chunker.chunk(text)
        chunk_embeddings = self.embedder.batch_embed(chunks)
        return self.pooler.mean_pool(chunk_embeddings)

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]