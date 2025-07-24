# stephanie/agents/maintenance/mrq_trainer_agent.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.training.mrq_trainer import MRQTrainer
from stephanie.scoring.scorable_factory import ScorableFactory, TargetType

class MRQTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.trainer = MRQTrainer(cfg, memory=memory, logger=logger)

    def _extract_samples(self, context):
        goal = context.get("goal", {})
        documents = context.get("documents", [])
        samples = []
        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            score = self.memory.scores.get_score(goal_id=goal["id"], scorable_id=scorable.id)
            if score:
                samples.append({
                    "title": goal.get("goal_text", ""),
                    "output": scorable.text,
                    "score": score.score
                })
        return samples

    async def run(self, context: dict) -> dict:
        """
        Agent entry point to train MRQ models for all configured dimensions.
        """
        results = {}
        for dim in self.trainer.dimensions:
            samples = self._extract_samples(context)
            if not samples:
                self.logger.log("NoSamplesFound", {"dimension": dim})
                continue
            stats = self.trainer.train(samples, dim)
            if "error" not in stats:
                results[dim] = stats

        context["training_stats"] = results
        return context
