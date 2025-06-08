from co_ai.analysis.score_evaluator import ScoreEvaluator
from co_ai.constants import HYPOTHESES, REVIEW


class ReviewScoringMixin:
    """
    A mixin that provides review scoring functionality to any agent.
    Can be used in ReviewAgent, MetaReviewAgent, or any composite evaluator agent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.review_scorer = None  # Will be initialized on first use


    def get_review_scorer(self):
        if not self.review_scorer:
            config_path = self.cfg.get("review_score_config", "config/scoring/review.yaml")
            self.review_scorer = ScoreEvaluator(
                self.cfg,
                self.prompt_loader,
                ScoreEvaluator.load_dimensions_from_config(config_path)
            )
        return self.review_scorer

    def score_hypothesis_with_review(self, hyp: dict, context: dict) -> dict:
        """
        Score a hypothesis using its review text.

        Args:
            hyp (dict): Hypothesis dictionary containing "text" and optionally "review".
            context (dict): Execution context for prompt generation or metadata.

        Returns:
            float: The computed score.
        """
        hyp_text = hyp.get("text")
        hyp_id = self.get_hypothesis_id(hyp)

        # Score the hypothesis based on the review
        scorer = self.get_review_scorer()

        score = scorer.evaluate(
            goal=context.get("goal", ""),
            hypothesis=hyp_text,
            context=context,
            llm_fn=self.call_llm
        )
        # Log the scoring event
        self.logger.log("ReviewScoreComputed", {"hypothesis_id": hyp_id, "score": score, "scores": scorer.scores})
        return {"id": hyp_id, "score": score, "scores": scorer.scores}


