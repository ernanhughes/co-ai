# co_ai/agents/proximity.py
import itertools

import numpy as np

from co_ai.agents.base import BaseAgent
from co_ai.constants import (DATABASE_MATCHES, GOAL, GOAL_TEXT,
                             PIPELINE_RUN_ID, TEXT)
from co_ai.models import ScoreORM
from co_ai.scoring.proximity import ProximityScore


class ProximityAgent(BaseAgent):
    """
    The Proximity Agent calculates similarity between hypotheses and builds a proximity graph.
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.similarity_threshold = cfg.get("similarity_threshold", 0.75)
        self.max_graft_candidates = cfg.get("max_graft_candidates", 3)
        self.top_k_database_matches = cfg.get("top_k_database_matches", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get(GOAL_TEXT)
        current_hypotheses = self.get_hypotheses(context)

        db_texts = self.memory.hypotheses.get_similar(goal_text, limit=self.top_k_database_matches)
        self.logger.log("DatabaseHypothesesMatched", {
            GOAL: goal,
            "matches": [{"text": h[:100]} for h in db_texts],
        })

        hypotheses_texts = [h.get(TEXT) for h in current_hypotheses]
        all_hypotheses = list(set(hypotheses_texts + db_texts))

        if not all_hypotheses:
            self.logger.log("NoHypothesesForProximity", {"reason": "empty_input"})
            return context

        similarities = self._compute_similarity_matrix(all_hypotheses)
        self.logger.log("ProximityGraphComputed", {
            "total_pairs": len(similarities),
            "threshold": self.similarity_threshold,
            "top_matches": [
                {"pair": (h1[:60], h2[:60]), "score": sim}
                for h1, h2, sim in similarities[:3]
            ],
        })

        graft_candidates = [(h1, h2) for h1, h2, sim in similarities if sim >= self.similarity_threshold]
        clusters = self._cluster_hypotheses(graft_candidates)

        context[self.output_key] = {
            "clusters": clusters,
            "graft_candidates": graft_candidates,
            DATABASE_MATCHES: db_texts,
            "proximity_graph": similarities,
        }


        top_similar = similarities[: self.max_graft_candidates]
        to_merge = {
                GOAL: goal,
                "most_similar": "\n".join(
                    [f"{i + 1}. {h1} ↔ {h2} (sim: {score:.2f})"
                     for i, (h1, h2, score) in enumerate(top_similar)]
                ),
            }

        merged = {**context, **to_merge}
        summary_prompt = self.prompt_loader.load_prompt(
            self.cfg, merged
        )

        summary_output = self.call_llm(summary_prompt, merged)
        context["proximity_summary"] = summary_output

        scorer = ProximityScore(self.cfg, memory=self.memory, logger=self.logger)
        score = scorer.compute({"proximity_analysis": summary_output}, context)

        self.logger.log(
            "ProximityAnalysisScored",
            {
                "score": score,
                "analysis": summary_output[:300],
            },
        )

        # Compute additional dimensions
        cluster_count = len(clusters)
        top_k_sims = [sim for _, _, sim in similarities[:self.max_graft_candidates]]
        avg_top_k_sim = sum(top_k_sims) / len(top_k_sims) if top_k_sims else 0.0
        graft_count = len(graft_candidates)

        dimensions = {
            "proximity_score": score,
            "cluster_count": cluster_count,
            "avg_similarity_top_k": round(avg_top_k_sim, 4),
            "graft_pair_count": graft_count,
        }

        # Save dimensional score for each hypothesis
        for hypothesis in current_hypotheses:
            score_obj = ScoreORM(
                agent_name=self.name,
                model_name=self.model_name,
                goal_id=goal.get("goal_id"),
                hypothesis_id=hypothesis.get("id"),
                score_type=self.name,
                evaluator_name=self.name,
                score=score,
                extra_data={"summary": summary_output},
                dimensions=dimensions,  # ✅ Added here
                pipeline_run_id=context.get(PIPELINE_RUN_ID),
            )
            self.memory.scores.insert(score_obj)

        return context

    def _compute_similarity_matrix(self, hypotheses: list[str]) -> list[tuple]:
        vectors = []
        valid_hypotheses = []

        for h in hypotheses:
            vec = self.memory.embedding.get_or_create(h)
            if vec is None:
                self.logger.log("MissingEmbedding", {"hypothesis_snippet": h[:60]})
                continue
            vectors.append(vec)
            valid_hypotheses.append(h)

        similarities = []
        for i, j in itertools.combinations(range(len(valid_hypotheses)), 2):
            h1 = valid_hypotheses[i]
            h2 = valid_hypotheses[j]
            sim = self._cosine(vectors[i], vectors[j])
            similarities.append((h1, h2, sim))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities

    def _cosine(self, a, b):
        a = np.array(list(a), dtype=float)
        b = np.array(list(b), dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _cluster_hypotheses(self, graft_candidates: list[tuple]) -> list[list[str]]:
        clusters = []

        for h1, h2 in graft_candidates:
            found = False
            for cluster in clusters:
                if h1 in cluster or h2 in cluster:
                    if h1 not in cluster:
                        cluster.append(h1)
                    if h2 not in cluster:
                        cluster.append(h2)
                    found = True
                    break
            if not found:
                clusters.append([h1, h2])

        merged_clusters = []
        for cluster in clusters:
            merged = False
            for mc in merged_clusters:
                if set(cluster) & set(mc):
                    mc.extend(cluster)
                    merged = True
                    break
            if not merged:
                merged_clusters.append(list(set(cluster)))

        return merged_clusters
