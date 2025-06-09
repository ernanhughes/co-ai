import os
import pickle
import numpy as np
from collections import defaultdict

from co_ai.agents.base import BaseAgent
from co_ai.models.unified_mrq import UnifiedMRQModelORM
from co_ai.evaluator.mrq_trainer import MRQTrainer
from co_ai.tools.cos_sim_tool import cosine_similarity


class UnifiedMRQAgent(BaseAgent):
    """
    Unified Multidimensional MR.Q Agent
    - Collects scores across all pipelines and dimensions.
    - Builds contrastive training pairs.
    - Trains a multidimensional preference model.
    - Saves models and logs metadata to DB.
    """

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.target_dimensions = cfg.get(
            "target_dimensions", ["correctness", "originality", "clarity", "relevance"]
        )
        self.similarity_threshold = cfg.get("similarity_threshold", 0.85)
        self.top_k_similar = cfg.get("top_k_similar", 20)
        self.min_score_difference = cfg.get("min_score_difference", 10)
        self.output_dir = cfg.get("model_output_dir", "mrq_models")
        self.trainer = MRQTrainer(memory, logger)

    async def run(self, context: dict) -> dict:
        self.logger.log("UnifiedMRQStarted", {})

        # Step 1: Load hypotheses and scores
        hypotheses = context.get("hypotheses") or self.memory.hypotheses.get_all()
        scores = self.memory.evaluations.get_all()

        # Step 2: Embed and index hypotheses
        embedded = self._index_embeddings(hypotheses)

        # Step 3: Collect dimension-wise scores
        score_map = self._group_scores(scores)

        # Step 4: Generate contrast pairs
        contrast_pairs = self._generate_contrast_pairs(embedded, score_map)

        # Step 5: Train model per dimension
        trained_models = self.trainer.train_multidimensional_model(contrast_pairs)
        self.logger.log(
            "UnifiedMRQTrained",
            {
                "pair_count": len(contrast_pairs),
                "dimensions": list(trained_models.keys()),
            },
        )

        # Step 6: Save and log to DB
        os.makedirs(self.output_dir, exist_ok=True)
        for dim, model in trained_models.items():
            path = os.path.join(self.output_dir, f"{dim}_mrq.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)

            self.memory.session.add(
                UnifiedMRQModelORM(
                    dimension=dim,
                    model_path=path,
                    pair_count=len([p for p in contrast_pairs if p[2] == dim]),
                    trainer_version="v1.0",
                    context={
                        "similarity_threshold": self.similarity_threshold,
                        "min_score_diff": self.min_score_difference,
                    },
                )
            )

        self.memory.session.commit()
        self.logger.log(
            "UnifiedMRQModelsSaved", {"dimensions": list(trained_models.keys())}
        )
        context["unified_mrq_model_paths"] = {
            dim: os.path.join(self.output_dir, f"{dim}_mrq.pkl")
            for dim in trained_models
        }

        return context

    def _index_embeddings(self, hypotheses):
        index = {}
        for hyp in hypotheses:
            emb_id = hyp.get("embedding_id")
            vector = hyp.get("embedding_vector")  # optional shortcut

            if vector is not None:
                index[hyp["id"]] = (hyp, np.array(vector))
            elif emb_id:
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
            for id2, (hyp2, vec2) in items[i + 1 :]:
                sim = cosine_similarity(vec1, vec2)
                if sim < self.similarity_threshold:
                    continue
                for dim in self.target_dimensions:
                    s1 = score_map.get(id1, {}).get(dim)
                    s2 = score_map.get(id2, {}).get(dim)
                    if (
                        s1 is not None
                        and s2 is not None
                        and abs(s1 - s2) >= self.min_score_difference
                    ):
                        better = hyp1 if s1 > s2 else hyp2
                        worse = hyp2 if s1 > s2 else hyp1
                        pairs.append((better["text"], worse["text"], dim))
        return pairs
