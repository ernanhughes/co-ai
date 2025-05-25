from co_ai.agents.base import BaseAgent

class IdeaInnovationAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {}).get("goal_text", "")
        survey = context.get("survey_results", {})
        arxiv_titles = [entry["title"] for entry in survey.get("arxiv", [])]
        dataset_names = [entry["name"] for entry in survey.get("huggingface", [])]

        input_vars = {
            "goal": goal,
            "arxiv_titles": arxiv_titles,
            "datasets": dataset_names
        }

        prompt = self.prompt.render(input_vars)
        self.logger.log("IdeaPromptGenerated", {"prompt": prompt})

        idea = await self.call_llm(prompt)

        context["idea"] = idea
        self.logger.log("IdeaGenerated", {"idea": idea})
        return context
