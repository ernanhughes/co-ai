import statistics
from dataclasses import dataclass

from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.mixins.scoring_mixin import ScoringMixin
from co_ai.constants import PIPELINE_RUN_ID
from co_ai.models.rule_application import RuleApplicationORM
from co_ai.models.symbolic_rule import SymbolicRuleORM


@dataclass
class RuleMutationAgentConfig:
    prompt_file: str = "rule_mutation.j2"
    min_applications: int = 3
    min_score_threshold: float = 6.5
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 512


class RuleMutationAgent(ScoringMixin, BaseAgent):
    def __init__(self, cfg, memory=None, logger=None, config: RuleMutationAgentConfig = RuleMutationAgentConfig()):
        super().__init__(cfg, memory, logger)
        self.config = config

    async def run(self, context: dict) -> dict:
        self.logger.log("RuleMutationStart", {"run_id": context.get(PIPELINE_RUN_ID)})

        rule_apps = self.memory.session.query(RuleApplicationORM).all()
        grouped = self._group_by_rule(rule_apps)

        for rule_id, applications in grouped.items():
            if len(applications) < self.config.min_applications:
                continue

            scores = [app.result_score for app in applications if app.result_score is not None]
            if not scores or statistics.mean(scores) >= self.config.min_score_threshold:
                continue  # Skip well-performing rules

            rule = self.memory.session.query(SymbolicRuleORM).get(rule_id)
            feedback = "\n".join([app.explanation or "" for app in applications[:3]])

            prompt = self.prompt_loader.from_file(
                self.config.prompt_file,
                config=self.cfg,
                context={
                    "rule": rule.to_dict(),
                    "feedback": feedback,
                    "scores": scores[:5]
                },
            )

            self.logger.log("MutationPromptBuilt", {"rule_id": rule_id})
            response = self.call_llm(prompt, context)
            self.logger.log("RuleMutated", {"rule_id": rule_id, "mutation": response[:150]})

            # You could optionally save the mutation as a new rule
            # self._save_mutated_rule(rule, response.strip())

        self.logger.log("RuleMutationEnd", {"run_id": context.get(PIPELINE_RUN_ID)})
        return context

    def _group_by_rule(self, rule_apps):
        grouped = {}
        for app in rule_apps:
            grouped.setdefault(app.rule_id, []).append(app)
        return grouped
