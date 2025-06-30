import torch
from co_ai.agents.base_agent import BaseAgent
from co_ai.scoring.document_mrq_scorer import DocumentMRQScorer
from co_ai.evaluator.text_encoder import TextEncoder
from co_ai.scoring.document_value_predictor import DocumentValuePredictor


class DocumentRewardScorerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dimensions = cfg.get("dimensions", ["relevance", "clarity", "engagement"])
        self.model_dir = cfg.get("model_dir", "models/document")
        self.model_prefix = cfg.get("model_prefix", "document_rm_")

        self.encoder = TextEncoder().to(self.device)
        self.value_predictor = DocumentValuePredictor().to(self.device)

        # Load MRQ Scorer
        self.scorer = DocumentMRQScorer(
            memory=self.memory,
            logger=self.logger,
            dimensions=self.dimensions,
            model_dir=self.model_dir,
            model_prefix=self.model_prefix,
            device=self.device
        )

    async def run(self, context: dict) -> dict:
        documents = context.get("documents", [])
        goal = context.get("goal", {})
        goal_text = goal.get("goal_text", "")
        results = []

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            if not content:
                continue

            scores = {}
            for dim in self.dimensions:
                score = self.scorer.score(goal_text, content, dimension=dim)
                scores[dim] = score

            results.append({
                "title": title,
                "text": content,
                "scores": scores
            })

        context[self.output_key] = results
        return context
