# co_ai/agents/seal/rule_mutation_agent.py

import statistics
from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.models.rule_application import RuleApplicationORM
from co_ai.models.symbolic_rule import SymbolicRuleORM
from co_ai.constants import PIPELINE_RUN_ID


class RuleMutationAgent(ScoringMixin, BaseAgent):
    def __init__(self, *args, min_applications=3, min_score_threshold=6.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_applications = min_applications
        self.min_score_threshold = min_score_threshold

    async def run(self, context: dict) -> dict:
        self.logger.log("RuleMutationStart", {"run_id": context.get(PIPELINE_RUN_ID)})

        rule_apps = self.memory.session.query(RuleApplicationORM).all()
        grouped = self._group_by_rule(rule_apps)

        for rule_id, applications in grouped.items():
            if len(applications) < self.min_applications:
                continue

            scores = [app.result_score for app in applications if app.result_score is not None]
            if not scores:
                continue

            avg_score = statistics.mean(scores)
            if avg_score >= self.min_score_threshold:
                continue  # Skip good rules

            rule = self.memory.session.query(SymbolicRuleORM).get(rule_id)
            feedback = self._summarize_failures(applications)
            context_vars = {
                "rule": {
                    "name": rule.name,
                    "description": rule.description,
                    "condition": rule.condition,
                    "action": rule.action
                },
                "feedback": feedback,
                "score_deltas": scores,
            }

            prompt = self.prompt_loader.from_file("rule_eval_prompt.j2", context=context_vars)
            response = await self.call_llm(prompt, context)
            self.logger.log("RuleMutated", {
                "rule_id": rule_id,
                "response": response.strip(),
                "avg_score": avg_score
            })

            # Optional: Score or save the mutated rule
            # score = self.score_hypothesis(...)

        self.logger.log("RuleMutationEnd", {"run_id": context.get(PIPELINE_RUN_ID)})
        return context

    def _group_by_rule(self, rule_apps):
        grouped = {}
        for app in rule_apps:
            grouped.setdefault(app.rule_id, []).append(app)
        return grouped

    def _summarize_failures(self, applications):
        reasons = [app.failure_reason for app in applications if app.failure_reason]
        if not reasons:
            return "No specific failure feedback available."
        return "\n".join(f"- {r}" for r in reasons[:5])
