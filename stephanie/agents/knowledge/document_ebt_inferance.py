import os

import torch
from torch import nn

from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mrq.mrq_scorer import MRQScorer  # Assumed
from stephanie.memory.embedding_store import EmbeddingStore
from stephanie.scoring.document_ebt_trainer import DocumentEBTScorer


class DocumentEBTInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_save_path", "models")
        self.model_prefix = cfg.get("model_prefix", "document_ebt_")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = memory.embedding

        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "novelty", "relevance", "implementability"])
        self.models = {}
        for dim in self.dimensions:
            model = DocumentEBTScorer().to(self.device)
            path = os.path.join(self.model_path, f"{self.model_prefix}{dim}.pt")
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.models[dim] = model

    def score_document(self, context_text: str, document_text: str) -> dict:
        ctx_emb = torch.tensor(self.embedding.get_or_create(context_text)).unsqueeze(0).to(self.device)
        doc_emb = torch.tensor(self.embedding.get_or_create(document_text)).unsqueeze(0).to(self.device)

        scores = {}
        for dim, model in self.models.items():
            with torch.no_grad():
                score = model(ctx_emb, doc_emb).item()
                scores[dim] = round(score, 3)
        return scores

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        document = context.get("document")
        context[self.output_key] = self.score_document(goal_text, document)
        return context
