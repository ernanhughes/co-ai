from co_ai.analysis.score_evaluator import ScoreEvaluator
from co_ai.constants import HYPOTHESES, REVIEW

class ReviewScoringMixin:
    """
    A mixin that provides review scoring functionality to any agent.
    Designed for use with ScoreEvaluator using multi-dimensional scoring.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_scorer = None  # Lazy init

    def get_review_scorer(self):
        if self.review_scorer is None:
            config_path = self.cfg.get("review_score_config", "config/scoring/review.yaml")
            self.review_scorer = ScoreEvaluator.from_file(
                filepath=config_path,
                prompt_loader=self.prompt_loader,
                agent_config=self.cfg
            )
        return self.review_scorer

    def score_hypothesis_with_review(self, hyp: dict, context: dict) -> dict:
        """
        Score a hypothesis using the configured multi-dimensional evaluator.

        Args:
            hyp (dict): Hypothesis dictionary (must contain 'text').
            context (dict): Pipeline context (should contain 'goal').

        Returns:
            dict: {
                "id": hypothesis_id,
                "score": float (weighted average),
                "scores": {dim_name: {score, rationale, weight}, ...}
            }
        """
        hypothesis_id = self.get_hypothesis_id(hyp)
        hypothesis_text = hyp.get("text")
        goal = context.get("goal", "")

        scorer = self.get_review_scorer()
        dimension_scores = scorer.evaluate(
            goal=goal,
            hypothesis=hypothesis_text,
            context=context,
            llm_fn=self.call_llm
        )

        weighted_total = sum(
            s["score"] * s.get("weight", 1.0)
            for s in dimension_scores.values()
        )
        weight_sum = sum(s.get("weight", 1.0) for s in dimension_scores.values())
        final_score = round(weighted_total / weight_sum, 2) if weight_sum > 0 else 0.0

        self.logger.log("HypothesisScoreComputed", {
            "hypothesis_id": hypothesis_id,
            "score": final_score,
            "dimension_scores": dimension_scores
        })

        return {
            "id": hypothesis_id,
            "score": final_score,
            "scores": dimension_scores
        }
