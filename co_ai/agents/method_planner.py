from co_ai.agents.base import BaseAgent

class MethodDevelopmentAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        idea = context.get("idea", "")
        baseline_code = context.get("baseline_code", "")
        literature = context.get("survey_results", {}).get("arxiv", [])

        # Extract core hypothesis and variables from idea
        prompt = self.prompt.render({
            "idea": idea,
            "baseline_code": baseline_code,
            "literature": "\n".join([paper["title"] for paper in literature])
        })

        method_plan = await self.call_llm(prompt)
        context["method_plan"] = method_plan
        self.logger.log("MethodPlanGenerated", {"plan": method_plan})
        return context