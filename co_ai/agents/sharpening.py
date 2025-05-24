# co_ai/agents/sharpening.py

from dataclasses import asdict

from co_ai.agents import BaseAgent
from co_ai.constants import GOAL, PIPELINE
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.models import Hypothesis
from co_ai.models.sharpening_result import SharpeningResult


class SharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, logger, device=self.device)
        self.templates = cfg.get("templates", ["critic"])

    async def run(self, context: dict):
        goal = self.extract_goal_text(context.get(GOAL))

        self.evaluator.train_from_database(goal=goal, cfg=self.cfg)

        prompts = context.get("prompt_history", {}).get(self.target, [])
        results = []
        for data in prompts:
            if self.cfg.get("mode", "template") == "judge":
                result = self.run_judge_only(data, context)
            elif self.cfg.get("mode", "template") == "compare_mrq":
                result = self.compare_mrq(data, context)
            else:
                result = self.run_selected(data, context)
            results.append(result)
            if self.cfg.get("log_results", False):
                self.log_sharpening_results(goal, data.get("prompt"), data.get("response"), result)
        context[self.output_key] = results
        return context

    def run_selected(self, data: dict, context: dict) -> list[dict]:
        goal = self.extract_goal_text(context.get(GOAL))
        results = []
        prompt = data.get("prompt")
        examples = self.memory.hypotheses.get_hypotheses_for_prompt(prompt, 3)
        merged = {**context, **{"prompt": prompt, "examples": examples}}

        if prompt:
            for name in self.templates:
                prompt_template = self.prompt_loader.from_file(name, self.cfg, merged)
                sharpened_hypothesis = self.call_llm(prompt_template, merged)
                hypothesis = data.get("response") # hypotheses result for prompt
                preferred_output, scores = self.evaluator.judge(goal,
                                                                prompt, hypothesis, sharpened_hypothesis
                                                                )
                value_a = scores["value_a"]
                value_b = scores["value_b"]
                winner = "a" if value_a >= value_b else "b"
                score = max(value_a, value_b)
                score_diff = abs(value_a - value_b)
                improved = winner == "b"
                comparison = "sharpened_better" if improved else "original_better"
                result = {
                    "template": name,
                    "winner": winner,
                    "improved": improved,
                    "comparison": comparison,
                    "score": round(score, 4),
                    "score_diff": round(score_diff, 4),
                    "output": preferred_output,
                    "raw_scores": scores,
                    "sharpened_hypothesis": sharpened_hypothesis,
                    "prompt_template": prompt_template,
                    PIPELINE: context.get(PIPELINE),
                }
                self.save_improved(goal, prompt_template, result)
                results.append(result)
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def compare_mrq(self, data: dict, context: dict) -> list[dict]:
        goal = self.extract_goal_text(context.get(GOAL))
        prompt = data.get("prompt")
        hypothesis = data.get("response")

        # For judge-only, use a simple reflection-based transformation (or leave unchanged)
        sharpened_hypothesis = hypothesis  # no change, just self-judging

        _, scores = self.evaluator.judge(goal, prompt, hypothesis, sharpened_hypothesis)
        value_a = scores["value_a"]
        value_b = scores["value_b"]
        winner = "a" if value_a >= value_b else "b"
        score = max(value_a, value_b)
        score_diff = abs(value_a - value_b)
        improved = winner == "b"
        comparison = "sharpened_better" if improved else "original_better"

        result = {
            "template": "judge_only",
            "winner": winner,
            "improved": improved,
            "comparison": comparison,
            "score": round(score, 4),
            "score_diff": round(score_diff, 4),
            "output": hypothesis,
            "raw_scores": scores,
            "sharpened_hypothesis": sharpened_hypothesis,
            "prompt_template": None,
            PIPELINE: context.get(PIPELINE),
        }

        if improved:
            self.save_improved(goal, prompt, result)

        return [result]

    async def run_judge_only(self, data:dict, context: dict):
        prompt = data.get("prompt")
        examples = self.memory.hypotheses.get_hypotheses_for_prompt(prompt, 3)
        merged = {**context, **{"prompt": prompt, "examples": examples}}
        prompt_template = self.prompt_loader.from_file("self_reward.txt", self.cfg, merged)
        response = self.call_llm(prompt_template, context)
        return response

    def save_improved(self, goal, prompt: str, entry: dict):
        if entry["improved"] and self.cfg.get("save_improved", True):
            # Save refined prompt (optional – only if different enough)
            new_prompt_id = self.memory.prompt.save(
                goal=goal,
                agent_name=f"{self.name}_{entry['template']}",
                prompt_key="sharpening",
                prompt_text=prompt,
                response=entry["sharpened_hypothesis"],
                strategy=self.cfg.get("strategy", "default"),
                meta_data={
                    "original_prompt": prompt,
                    "template": entry["template"],
                    "score_improvement": entry["score_diff"],
                },
            )

            self.logger.log(
                "SharpenedGoalSaved",
                {
                    "prompt_text": prompt[:100],
                },
            )
            hyp = Hypothesis(
                goal=goal, 
                text=entry["sharpened_hypothesis"], 
                prompt=prompt,
                pipeline_signature=entry.get(PIPELINE)
                )
            # Save new hypothesis for that prompt
            self.memory.hypotheses.insert(hyp)

            self.logger.log(
                "SharpenedHypothesisSaved",
                {
                    "prompt_id": new_prompt_id,
                    "text_snippet": entry["sharpened_hypothesis"][:100],
                    "score": entry["score"],
                },
            )

    def log_sharpening_results(
        self,
        goal: str,
        prompt: str,
        original_output: str,
        results: list[dict]
    ):
        for entry in results:
            result = SharpeningResult(
                goal=goal,
                prompt=prompt,
                template=entry["template"],
                original_output=original_output,
                sharpened_output=entry["sharpened_hypothesis"],
                preferred_output=entry["output"],
                winner=entry["winner"],
                improved=entry["improved"],
                comparison=entry["comparison"],
                score_a=entry["raw_scores"]["value_a"],
                score_b=entry["raw_scores"]["value_b"],
                score_diff=entry["score_diff"],
                best_score=entry["score"],
                prompt_template=entry.get("prompt_template", None),
            )
            self.memory.mrq.insert_sharpening_result(result)
            self.logger.log("SharpeningResultSaved", asdict(result))
