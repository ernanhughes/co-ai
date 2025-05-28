from co_ai.agents import BaseAgent
from co_ai.judges.base_judge import BaseJudge
from co_ai.models.llm import call_llm
import random

class AdaptiveReasonerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.modes = ["adaptive", "instruction_guided", "consensus_guided"]
        self.mode = self.config.get("mode", "adaptive")
        self.format_list = ["direct", "short_cot", "code", "long_cot"]
        self.judge: BaseJudge = self.registry.get("judge") if self.registry else None

    def run(self, goal):
        if self.mode == "instruction_guided":
            format_name = self.config.get("format", "long_cot")
            return self._generate_with_format(goal, format_name)

        elif self.mode == "consensus_guided":
            return self._run_consensus_mode(goal)

        else:  # default to adaptive
            return self._run_adaptive_mode(goal)

    def _generate_with_format(self, goal, format_name):
        prompt = self.prompt_loader.render(f"format_{format_name}.j2", goal=goal)
        response = call_llm(prompt, config=self.config.model)
        return {
            "response": response,
            "format": format_name
        }

    def _run_consensus_mode(self, goal):
        outputs = {}
        for fmt in ["direct", "short_cot", "code"]:
            outputs[fmt] = self._generate_with_format(goal, fmt)["response"]

        # Check for consensus
        responses = list(outputs.values())
        if len(set(responses)) == 1:
            return {
                "response": responses[0],
                "format": "consensus-simple",
                "source_formats": list(outputs.keys())
            }
        else:
            return self._generate_with_format(goal, "long_cot")

    def _run_adaptive_mode(self, goal):
        scores = {}
        results = {}

        for fmt in self.format_list:
            result = self._generate_with_format(goal, fmt)
            results[fmt] = result
            if self.judge:
                scores[fmt] = self.judge.score(goal, result["response"])
            else:
                # fallback: favor shorter formats randomly if no judge
                scores[fmt] = random.uniform(0.5, 1.0) - 0.1 * self.format_list.index(fmt)

        best_format = max(scores, key=scores.get)
        return results[best_format]
