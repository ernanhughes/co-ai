# co_ai/scoring/review.py

from co_ai.scoring.base_score import BaseScore
from co_ai.models import HypothesisORM

class ReviewScore(BaseScore):
    name = "review"
    default_value = 0.0

    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)

    def compute(self, hypothesis: HypothesisORM, context: dict) -> float:
        review_text = hypothesis.review
        if not review_text:
            return self.default_value

        parsed = self.parse_review_block(review_text)
        try:
            return float(parsed.get("score", self.default_value))
        except ValueError:
            return self.default_value

    @staticmethod
    def parse_review_block(review_text: str) -> dict:
        parsed = {}
        for line in review_text.strip().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                parsed[key.strip().lower()] = value.strip()
        return parsed
