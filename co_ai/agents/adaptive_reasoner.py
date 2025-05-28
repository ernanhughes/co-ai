from co_ai.agents import BaseAgent
from co_ai.evaluator import LLMJudgeEvaluator, MRQSelfEvaluator
from co_ai.reasoning.arm import ARMReasoningSelfEvaluator  # New import

# Optional utility for detecting reasoning formats
from co_ai.reasoning.arm import detect_format

from co_ai.dataloaders.arm_to_mrq_dpo import ARMDataLoader

class AdaptiveReasonerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

        self.modes = ["adaptive", "instruction_guided", "consensus_guided"]
        self.mode = self.cfg.get("mode", "adaptive")
        self.format_list = self.cfg.get(
            "format_list", ["direct", "short_cot", "code", "long_cot"]
        )
        self.judge = ARMReasoningSelfEvaluator(memory, logger)

    async def run(self, context:dict):
        adapter = ARMDataLoader(dataset_name="aqua_rat", split="train")
        adapter.adapt(context)

        response = ""
        if self.mode == "instruction_guided":
            format_name = self.cfg.get("format", "long_cot")
            response = self._generate_with_format(format_name, context)
        elif self.mode == "consensus_guided":
            response = self._run_consensus_mode(context)
        else:  # default to adaptive
            response = self._run_adaptive_mode(context)

        context[self.output_key] = response
        return context

    def _generate_with_format(self, fmt, context):
        prompt = self.prompt_loader.from_file(fmt, self.cfg, context)
        response = self.call_llm(prompt, context)
        return {
            "prompt": prompt,
            "response": response,
            "format_used": detect_format(response) or fmt
        }

    def _run_consensus_mode(self, goal):
        outputs = {}
        for fmt in ["direct", "short_cot", "code"]:
            outputs[fmt] = self._generate_with_format(goal, fmt)["response"]

        responses = list(outputs.values())
        unique_responses = set(responses)

        if len(unique_responses) == 1:
            return {
                "response": responses[0],
                "format": "consensus-simple",
                "source_formats": list(outputs.keys()),
            }
        else:
            long_cot_response = self._generate_with_format(goal, "long_cot")
            return {
                "response": long_cot_response["response"],
                "format": "long_cot",
                "source_formats": list(outputs.keys()),
                "fallback_reason": "no_consensus"
            }

    def _run_adaptive_mode(self, context):
        scores = {}
        results = {}

        prioritized = self._get_prioritized_formats(context)
        format_usage_freq = {fmt: 0 for fmt in prioritized}

        for fmt in prioritized:
            result = self._generate_with_format(fmt, context)
            results[fmt] = result
            response = result["response"]

            # Base score from judge
            if isinstance(self.judge, (MRQSelfEvaluator, ARMReasoningSelfEvaluator)):
                base_score = self.judge.score(context, response)
            elif hasattr(self.judge, 'score'):
                base_score = self.judge.score(context, response)
            else:
                base_score = 1.0 - 0.05 * prioritized.index(fmt)

            # Token efficiency penalty
            token_len = len(response.split())
            token_efficiency_penalty = 0.01 * token_len

            # Format rarity bonus
            rarity_bonus = 1.0 / (1 + format_usage_freq.get(fmt, 0))

            # Total score
            final_score = base_score - token_efficiency_penalty + rarity_bonus

            scores[fmt] = final_score
            format_usage_freq[fmt] += 1

        best_format = max(scores, key=scores.get)
        chosen_result = results[best_format]

        # Log decision
        self.logger.log("AdaptiveModeDecision", {
            "goal": context,
            "scores": scores,
            "chosen": best_format
        })

        return chosen_result

    def get_format_for_goal(self, goal: dict):
        if "preferred_format" in goal:
            return goal["preferred_format"]
        goal_type = goal.get("goal_type", "default")
        if goal_type == "math":
            return "code"
        elif goal_type == "commonsense":
            return "short_cot"
        else:
            return "long_cot"

    def _get_prioritized_formats(self, context):
        if "preferred_format" in context:
            return [context["preferred_format"]]

        priority_map = self.cfg.get("format_priority_by_difficulty", {})
        difficulty = context.get("difficulty", "default").lower()
        return priority_map.get(difficulty, priority_map.get("default", ["long_cot"]))

    def _init_judge(self):
        judge_strategy = self.cfg.get("judge", "mrq")
        if judge_strategy == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("judge_prompt_file", "judge_pairwise_comparison.txt")
            self.logger.log("EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file})
            return LLMJudgeEvaluator(self.cfg, llm, prompt_file, self.call_llm, self.logger)

        elif judge_strategy == "arm":
            self.logger.log("EvaluatorInit", {"strategy": "ARM"})
            return ARMReasoningSelfEvaluator(self.memory, self.logger, device="cuda")

        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)

class MyAdaptiveAgent(ARMReasoningSelfEvaluator):
    def __init__(self, model, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer

    def _generate_with_format(self, prompt: str, fmt: str) -> str:
        """
        Actually generate response using the specified format.
        Wraps the prompt with format tokens before calling the model.
        """
        format_prefixes = {
            "direct": "<Direct>",
            "short_cot": "<Short_CoT>",
            "code": "<Code>",
            "long_cot": "<Long_CoT>"
        }

        full_prompt = f"{format_prefixes.get(fmt, '')}{prompt}"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
        return response