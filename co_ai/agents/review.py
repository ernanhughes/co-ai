# co_ai/agents/review.py
from co_ai.agents.base import BaseAgent
from co_ai.constants import HYPOTHESES, REVIEW
from co_ai.models import ScoreORM
from co_ai.scoring.review import ReviewScore

class ReviewAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        hypotheses = self.get_hypotheses(context)
        reviews = []
        review_scorer = ReviewScore(self.cfg, memory=self.memory, logger=self.logger)
        for h in hypotheses:
            prompt = self.prompt_loader.load_prompt(self.cfg, {**context, **{HYPOTHESES:h.text}})
            review = self.call_llm(prompt, context)
            self.memory.hypotheses.update_review(h.id, review)
            score = review_scorer.get_score(h, context)
            reviews.append(review)
            self.logger.log(
                "ReviewScoreComputed", {"hypothesis_id": h.id, "score": score, "review": review}
            )

        context[self.output_key] = reviews
        return context
