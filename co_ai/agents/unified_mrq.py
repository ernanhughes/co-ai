# co_ai/agents/unified_mrq_agent.py

import numpy as np
from co_ai.agents.base import BaseAgent
from co_ai.models import HypothesisORM, ScoreORM, EmbeddingORM
from co_ai.evaluator.mrq_trainer import MRQTrainer
from co_ai.tools.cos_sim_tool import cosine_similarity
from collections import defaultdict


class UnifiedMRQAgent(BaseAgent):
    """
    Unified Multidimensional MR.Q Agent
    - Collects scores across all pipelines and dimensions.
    - Builds contrastive training pairs.
    - Trains a multidimensional preference model.
    - Returns inference-quality judgments on new hypotheses.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.target_dimensions = cfg.get("target_dimensions", ["correctness", "originality", "clarity", "relevance"])
        self.similarity_threshold = cfg.get("similarity_threshold", 0.85)
        self.top_k_similar = cfg.get("top_k_similar", 20)
        self.min_score_diff = cfg.get("min_score_diff", 10)
        self.trainer = MRQTrainer(cfg, logger)

    async def run(self, context: dict) -> dict:
        self.logger.log("UnifiedMRQStarted", {})

        # Load all hypotheses and scores
        hypotheses = self.memory.hypotheses.get_all()
        scores = self.memory.scores.get_all()

        # Step 1: Embed and index hypotheses
        embedded = self._index_embeddings(hypotheses)

        # Step 2: Collect dimension-wise scores
        score_map = self._group_scores(scores)

        # Step 3: Generate contrast pairs per dimension
        contrast_pairs = self._generate_contrast_pairs(embedded, score_map)

        # Step 4: Train the unified MR.Q model
        model = self.trainer.train_multidimensional_model(contrast_pairs)
        self.logger.log("UnifiedMRQTrained", {"pair_count": len(contrast_pairs)})

        context["unified_mrq_model"] = model
        return context

    def _index_embeddings(self, hypotheses):
        index = {}
        for hyp in hypotheses:
            emb_id = hyp.get("embedding_id")
            embedding = self.memory.embeddings.get_by_id(emb_id)
            if embedding:
                index[hyp["id"]] = (hyp, np.array(embedding["vector"]))
        return index

    def _group_scores(self, scores):
        grouped = defaultdict(dict)
        for s in scores:
            grouped[s.hypothesis_id][s.dimension_name] = s.score
        return grouped

    def _generate_contrast_pairs(self, embedded, score_map):
        pairs = []
        items = list(embedded.items())
        for i, (id1, (hyp1, vec1)) in enumerate(items):
            for id2, (hyp2, vec2) in items[i + 1:]:
                sim = cosine_similarity(vec1, vec2)
                if sim < self.similarity_threshold:
                    continue
                for dim in self.target_dimensions:
                    s1 = score_map.get(id1, {}).get(dim)
                    s2 = score_map.get(id2, {}).get(dim)
                    if s1 is not None and s2 is not None and abs(s1 - s2) >= self.min_score_diff:
                        better = hyp1 if s1 > s2 else hyp2
                        worse = hyp2 if s1 > s2 else hyp1
                        pairs.append((better["text"], worse["text"], dim))
        return pairs
