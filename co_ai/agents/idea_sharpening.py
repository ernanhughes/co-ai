# co_ai/agents/idea_sharpening.py
from dataclasses import asdict

from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL, PIPELINE
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.models import HypothesisORM
from co_ai.models.sharpening_result import SharpeningResult


class IdeaSharpeningAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.target = cfg.get("target", "generation")
        self.device = cfg.get("device", "cpu")
        self.evaluator = MRQSelfEvaluator(memory, logger, device=self.device)
        self.templates = cfg.get("templates", ["critic"])
        self.output_key = cfg.get("output_key", "sharpened_ideas")


    async def run(self, context: dict) -> dict:
        """
        Main execution loop for IdeaSharpeningAgent.

        Takes a list of ideas, sharpens them using templates,
        judges against baseline using evaluator, and logs results.
        """
        goal = context.get(GOAL, {})
        ideas = context.get("ideas", [])

        if not ideas:
            self.logger.log("NoIdeasToSharpen", {"reason": "empty_input"})
            return context

        sharpened_results = []
        for idea in ideas:
            result = await self._sharpen_and_evaluate(idea, goal, context)
            sharpened_results.append(result)

        # Sort by score
        sharpened_results.sort(key=lambda x: x["score"], reverse=True)

        # Update context
        context["sharpened_ideas"] = [r["sharpened_hypothesis"] for r in sharpened_results]
        context["scored_ideas"] = sharpened_results
        context["top_idea"] = sharpened_results[0]["sharpened_hypothesis"]

        return context

    async def _sharpen_and_evaluate(self, idea: str, goal: dict, context: dict) -> dict:
        # Build prompt for refinement
        merged = {
            "goal": goal,
            "idea": idea,
            "baseline": self.memory.baseline.get(goal.get("focus_area")),
            "literature_summary": context.get("knowledge_base_summaries", []),
            "examples": self.memory.hypotheses.get_hypotheses_for_prompt(idea, limit=3),
            "strategy": goal.get("strategy", "default"),
        }

        improved = None
        winner = "original"
        scores = {}

        for name in self.templates:
            prompt_template = self.prompt_loader.from_file(name, self.cfg, merged)
            sharpened = self.call_llm(prompt_template, merged)

            try:
                preferred_output, scores = self.evaluator.judge(
                    goal=goal.get("goal_text"),
                    prompt=idea,
                    output_a=idea,
                    output_b=sharpened,
                )
                improved = preferred_output
                winner = "b" if improved == sharpened else "a"
            except Exception as e:
                self.logger.log("IdeaSharpeningFailed", {"error": str(e)})
                improved = idea
                winner = "a"
                scores = {"value_a": 5.0, "value_b": 5.0}

            result = {
                "template_used": name,
                "original_idea": idea,
                "sharpened_hypothesis": improved,
                "winner": winner,
                "improved": winner == "b",
                "scores": scores,
                "score": max(scores.values()),
                "pipeline_stage": context.get(PIPELINE),
                "prompt_template": prompt_template,
            }

            self.save_improved(goal, idea, result, context)
            return result

    def save_improved(self, goal: dict, original_idea: str, result: dict, context: dict):
        if not result["improved"]:
            return

        # Save to Prompt ORM (optional)
        new_prompt_id = self.memory.prompt.save(
            goal=goal.get("goal_text"),
            agent_name=f"{self.name}_{result['template_used']}",
            prompt_key="idea_sharpening",
            prompt_text=original_idea,
            response=result["sharpened_hypothesis"],
            strategy=goal.get("strategy", "default"),
            meta_data={
                "score_diff": result["score_diff"],
                "prompt_template": result["prompt_template"],
            },
        )

        # Save to HypothesisORM
        hyp = HypothesisORM(
            goal=goal.get("goal_text"),
            text=result["sharpened_hypothesis"],
            prompt=original_idea,
            pipeline_signature=context.get(PIPELINE),
            origin="idea_sharpening_agent",
        )
        self.memory.hypotheses.insert(hyp)

        self.logger.log(
            "IdeaSharpenedAndSaved",
            {
                "prompt_snippet": original_idea[:100],
                "response_snippet": result["sharpened_hypothesis"][:100],
                "score": result["score"],
            },
        )