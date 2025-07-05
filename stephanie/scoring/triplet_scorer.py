# stephanie/scorers/triplet_scorer.py

from stephanie.models.evaluation import TargetType
from stephanie.scoring.scorable import Scorable


class TripletScorer:
    def __init__(self, scorer, logger=None):
        """
        Args:
            scorer: ScoringMixin-compatible object with a `score_item(scorable, context, metrics)` method
            logger: Optional logger
        """
        self.scorer = scorer
        self.logger = logger

    def score_triplet(self, triplet_orm, goal, context, metrics="cartridge") -> dict:
        """
        Scores a single triplet using the provided goal and metrics.

        Args:
            triplet_orm: A CartridgeTripleORM instance
            goal: The goal dict or ORM object (must contain 'goal_text')
            context: Additional runtime context for the scoring call
            metrics: Scoring dimension or list of dimensions

        Returns:
            A score result dictionary
        """
        try:
            triplet_text = f"({triplet_orm.subject}, {triplet_orm.predicate}, {triplet_orm.object})"
            merged_context = {
                "triplet": triplet_orm.to_dict(),
                "goal": goal,
                **context
            }

            scorable = Scorable(
                id=triplet_orm.id,
                text=triplet_text,
                target_type=TargetType.CARTRIDGE_TRIPLE,
            )

            score = self.scorer.score_item(scorable, merged_context, metrics=metrics)

            if self.logger:
                self.logger.log("TripletScored", score.to_dict())

            return score.to_dict()

        except Exception as e:
            if self.logger:
                self.logger.log("TripletScoringFailed", {"triplet_id": triplet_orm.id, "error": str(e)})
            return {}
