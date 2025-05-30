from abc import ABC, abstractmethod

from co_ai.models import ScoreORM


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
    def compute(self, hypothesis: dict, context:dict) -> float:
        pass

    def get_score(self, hypothesis: dict, context: dict) -> float:
        # 1. If already cached on object
        if hypothesis.get(f"{self.name}_score"):
            return hypothesis[f"{self.name}_score"]

        # 2. Compute and attach
        score = self.compute(hypothesis, context)
        hypothesis[f"{self.name}_score"] = score

        # 3. Store in scores table
        if self.memory:
            s = ScoreORM(
                goal_id=hypothesis.get("goal_id"),
                hypothesis_id=hypothesis.get("id"),
                agent_name=self.agent_name,
                model_name=self.model_name,
                evaluator_name=self.evaluator_name,
                score_type=self.name,
                score=score,
                pipeline_run_id=context.get("pipeline_run_id"),
            )
            try:
                self.memory.scores.insert(s)
                self.memory.commit()  # Ensure session commit happens
            except Exception as e:
                self.memory.refresh_session()
                score = self.default_value
                self.logger.log("ScoreInsertFailed", {"error": str(e)})

        # 4. Log
        self.logger.log("ScoreComputed", {
            "type": self.name,
            "score": score,
            "hypothesis_id": hypothesis.get("id")
        })

        return score
