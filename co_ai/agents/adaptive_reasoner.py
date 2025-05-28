from co_ai.agents import BaseAgent
from co_ai.evaluator import LLMJudgeEvaluator, MRQSelfEvaluator


class AdaptiveReasonerAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.modes = ["adaptive", "instruction_guided", "consensus_guided"]
        self.mode = self.cfg.get("mode", "adaptive")
        self.format_list = self.cfg.get(
            "format_list", ["direct", "short_cot", "code", "long_cot"]
        )
        self.judge = self._init_judge()

    def run(self, goal):
        if self.mode == "instruction_guided":
            format_name = self.config.get("format", "long_cot")
            return self._generate_with_format(goal, format_name)

        elif self.mode == "consensus_guided":
            return self._run_consensus_mode(goal)

        else:  # default to adaptive
            return self._run_adaptive_mode(goal)

    def _generate_with_format(self, format_name, context):
        prompt = self.prompt_loader.from_file(
            f"format_{format_name}.j2", self.cfg, context
        )
        response = self.call_llm(self.cfg, prompt=prompt, format=format_name)

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
                "source_formats": list(outputs.keys()),
            }
        else:
            return self._generate_with_format(goal, "long_cot")

    def _run_adaptive_mode(self, goal):
        scores = {}
        results = {}

        prioritized = self._get_prioritized_formats(goal)

        for fmt in prioritized:
            result = self._generate_with_format(goal, fmt)
            results[fmt] = result

            if self.judge:
                scores[fmt] = self.judge.score(goal, result["response"])
            else:
                # Prefer earlier formats slightly
                scores[fmt] = 1.0 - 0.05 * prioritized.index(fmt)

        best_format = max(scores, key=scores.get)
        return results[best_format]

    def get_format_for_goal(self, goal: dict):
        if hasattr(goal, "preferred_format"):
            return goal.preferred_format
        goal_type = goal.get("goal_type", "default")
        if goal_type == "math":
            return "code"
        elif goal_type == "commonsense":
            return "short_cot"
        else:
            return "long_cot"

    def _get_prioritized_formats(self, goal):
        # Prefer explicit format if provided
        if "preferred_format" in goal:
            return [goal["preferred_format"]]

        # Read config-defined format priorities
        priority_map = self.config.get("format_priority_by_difficulty", {})

        difficulty = goal.get("difficulty", "default").lower()
        formats = priority_map.get(
            difficulty, priority_map.get("default", ["long_cot"])
        )

        return formats

    def _init_judge(self):
        judge_strategy = self.cfg.get("judge", "mrq")
        if judge_strategy == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get(
                "judge_prompt_file", "judge_pairwise_comparison.txt"
            )
            self.logger.log(
                "EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file}
            )
            return LLMJudgeEvaluator(
                self.cfg, llm, prompt_file, self.call_llm, self.logger
            )
        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)
