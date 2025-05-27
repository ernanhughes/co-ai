from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL


class SurveyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        strategy = goal.get("strategy", self.strategy)
        survey_context = {
            "goal_text": goal.get("goal_text"),
            "focus_area": goal.get("focus_area"),
            "goal_type": goal.get("goal_type"),
            "strategy": goal.get("strategy"),
        }
        merged = {**context, **survey_context}
        prompt = self.prompt_loader.load_prompt(self.cfg, merged)
        queries = self.call_llm(prompt, context)
        self.logger.log(
            "SurveyQueriesGenerated", {"queries": queries, "strategy": strategy}
        )
        context[self.output_key] = self.expand_queries_to_goals(queries, goal)
        return context

    def expand_queries_to_goals(self, query_block: str, goal: dict):
        queries = [
            line.strip() for line in query_block.strip().splitlines() if line.strip()
        ]
        return [
            {
                "goal_text": q,
                "parent_goal": goal.get("goal_text"),
                "focus_area": goal.get("focus_area"),
                "strategy": goal.get("strategy"),
                "source": "survey_query",
            }
            for q in queries
        ]