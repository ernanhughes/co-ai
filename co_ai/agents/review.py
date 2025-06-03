from co_ai.agents.base import BaseAgent
from co_ai.agents.mixins.review_scoring_mixin import ReviewScoringMixin


class ReviewAgent(ReviewScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []

        for hyp in hypotheses:
            # Score and update review
            score = self.score_hypothesis_with_review(hyp, context)
            self.logger.log(
                "ReviewScoreComputed",
                score,
            )
            reviews.append(score)

        context[self.output_key] = reviews
        return context