# co_ai/compiler/scorer.py
from co_ai.scoring.mrq_scorer import MRQScorer
from co_ai.scoring.llm_scorer import LLMScorer

from co_ai.agents.compiler.reasoning_trace import ReasoningNode

class ReasoningNodeScorer:
    def __init__(self, scorer_type="mrq"):
        if scorer_type == "mrq":
            self.scorer = MRQScorer()
        else:
            self.scorer = LLMScorer()

    def score(self, node: ReasoningNode) -> float:
        return self.scorer.score(
            goal=node.goal,
            hypothesis={"text": node.response},
            dimensions=["correctness", "clarity", "depth"]
        )