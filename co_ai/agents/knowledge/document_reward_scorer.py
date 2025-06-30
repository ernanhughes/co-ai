# co_ai/agents/evaluation/document_reward_scorer.py

import torch
from co_ai.agents.base_agent import BaseAgent
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.scoring.document_value_predictor import DocumentValuePredictor


class DocumentRewardScorer(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["relevance", "clarity", "engagement"])
        self.model_dir = cfg.get("model_dir", "models")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = TextEncoder().to(self.device)
        self.models = {}

        for dim in self.dimensions:
            model_path = f"{self.model_dir}/document_rm_{dim}.pt"
            model = DocumentValuePredictor().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.models[dim] = model

    async def run(self, context: dict) -> dict:
        documents = context.get("documents", [])
        goal_text = context.get("goal", {}).get("goal_text", "")
        results = []

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")

            if not content:
                continue

            context_emb = torch.tensor(self.memory.embedding.get_or_create(goal_text)).unsqueeze(0).to(self.device)
            doc_emb = torch.tensor(self.memory.embedding.get_or_create(content)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                zsa = self.encoder(context_emb, doc_emb)
                scores = {
                    dim: self.models[dim](zsa).item() for dim in self.dimensions
                }

            doc_result = {
                "title": title,
                "scores": scores,
                "text": content,
            }
            results.append(doc_result)

        context[self.output_key] = results
        return context
