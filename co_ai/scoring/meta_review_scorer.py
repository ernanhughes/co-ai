from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.evaluator import LLMJudgeEvaluator
from co_ai.scoring.mrq_evaluator import MRQEvaluator


class MetaReviewScorer(ScoringMixin):
    def __init__(self, cfg: dict, memory, logger, fallback_to_llm=True):
        super().__init__(cfg, memory, logger)
        self.mrq_scorer = MRQEvaluator(memory, logger)
        self.llm_scorer = LLMJudgeEvaluator(memory, logger)
        self.use_llm_fallback = fallback_to_llm

    def score(self, goal, hypothesis, dimensions):
        # Try MR.Q scoring first
        mrq_scores = self.mrq_scorer.score(goal, hypothesis, dimensions)

        # If MR.Q returns None or is undertrained in a dimension...
        if self._needs_llm_fallback(mrq_scores):
            llm_scores = self.llm_scorer.score(goal, hypothesis, dimensions)
            # Optionally combine or defer to LLM
            combined = self._combine_scores(mrq_scores, llm_scores)
            return combined

        return mrq_scores
