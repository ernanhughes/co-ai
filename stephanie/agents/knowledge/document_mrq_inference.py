from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.mixins.scoring_mixin import ScoringMixin
from stephanie.scoring.mrq_scorer import MRQScorer
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class DocumentMRQInferenceAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        
        self.scorer = MRQScorer(cfg, memory=memory, logger=logger)
        self.scorer.load_models()

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {}).get("goal_text")

        results = []
        for doc in context.get(self.input_key, []):
            scorable = Scorable(id=doc.get("id"), text=doc.get("text", ""), target_type=TargetType.DOCUMENT)
            score_bundle = self.score_item(scorable, context, metrics="compiler", scorer=self.scorer)
            results.append({
                "scorable": scorable.to_dict(),
                "scores": score_bundle.to_dict()
            })

        context[self.output_key] = results
        return context
