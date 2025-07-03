from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.scoring.scorable import Scorable
from stephanie.models.evaluation import TargetType

class ReviewAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []

        for hyp in hypotheses:
            scorable = {
                "id": hyp.get("id", ""),
                "text": hyp.get("text", ""),
                "target_type": TargetType.HYPOTHESIS,
            }   

            # Score and update review
            score = self.score_item(scorable, context, metrics="review")
            self.logger.log(
                "ReviewScoreComputed",
                score,
            )
            reviews.append(score)

        context[self.output_key] = reviews
        return context