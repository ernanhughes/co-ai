# stephanie/scorers/cartridge_scorer.py

from stephanie.models.evaluation import TargetType
from stephanie.scoring.scorable import Scorable


class CartridgeScorer:
    def __init__(self, scorer, logger=None):
        """
        Args:
            scorer: Object that implements `score_item(scorable, context, metrics)`
            logger: Optional logger
        """
        self.scorer = scorer
        self.logger = logger

    def score_cartridge(self, cartridge_orm, goal, context, metrics="cartridge") -> dict:
        """
        Scores a full cartridge for alignment, usefulness, etc.

        Args:
            cartridge_orm: A CartridgeORM instance
            goal: Goal dict or ORM object containing 'goal_text'
            context: Runtime context
            metrics: A single dimension or a list of scoring metrics

        Returns:
            A dictionary representing the score result
        """
        try:
            merged_context = {
                "mode": "cartridge",
                "cartridge": cartridge_orm.to_dict(),
                "goal": goal,
                **context,
            }

            scorable = Scorable(
                id=cartridge_orm.id,
                text=cartridge_orm.markdown_content,
                target_type=TargetType.CARTRIDGE,
            )

            score = self.scorer.score_item(scorable, merged_context, metrics=metrics)

            if self.logger:
                self.logger.log("CartridgeScored", score.to_dict())

            return score.to_dict()

        except Exception as e:
            if self.logger:
                self.logger.log("CartridgeScoringFailed", {
                    "cartridge_id": getattr(cartridge_orm, "id", None),
                    "error": str(e)
                })
            return {}
