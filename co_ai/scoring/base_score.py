from abc import ABC, abstractmethod
from datetime import datetime

from co_ai.models import HypothesisORM,ScoreORM


class BaseScore(ABC):
    name: str = "unnamed"
    default_value: float = 0.0

    def __init__(self, cfg, memory, logger, evaluator_name=None):
        self.memory = memory
        self.logger = logger
        self.agent_name = cfg.get("name")
        self.model_name = cfg.get("model", {}).get("name")
        self.evaluator_name = evaluator_name or self.name

    @abstractmethod
    def compute(self, hypothesis: HypothesisORM, context:dict) -> float:
        pass

    def get_score(self, hypothesis: HypothesisORM, context: dict) -> float:
        # 1. If already cached on object
        if hasattr(hypothesis, f"{self.name}_score"):
            return getattr(hypothesis, f"{self.name}_score", self.default_value)

        # 2. Compute and attach
        score = self.compute(hypothesis, context)
        setattr(hypothesis, f"{self.name}_score", score)

        # 3. Store in scores table
        if self.memory:
            s = ScoreORM(goal_id=hypothesis.goal_id,
            hypothesis_id= hypothesis.id,
            agent_name=self.agent_name,
            model_name=self.model_name,
            evaluator_name=self.evaluator_name,
            score_type=self.name,
            score=score,
            run_id=context.get("run_id")
            )
            self.memory.scores.insert(s)

        # 4. Log
        self.logger.log("ScoreComputed", {
            "type": self.name,
            "score": score,
            "hypothesis_id": hypothesis.id
        })

        return score
