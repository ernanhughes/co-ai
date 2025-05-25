from co_ai.agents.base import BaseAgent
from co_ai.utils.query_generator import GoalQueryGenerator
from co_ai.tools import WebSearchTool
from co_ai.tools.arxiv_tool import search_arxiv
from co_ai.tools.huggingface_tool import search_huggingface_datasets

class SurveyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "default")
        self.query_generator = GoalQueryGenerator()
        self.web_search_tool = WebSearchTool(cfg.get("web_search", {}), self.logger)


    async def run(self, context: dict) -> dict:
        goal = context.get("goal", {})
        strategy = goal.get("strategy", self.strategy)
        queries = self.query_generator.generate_queries(goal, strategy)

        self.logger.log("SurveyQueriesGenerated", {"queries": queries, "strategy": strategy})

        results = {
            "arxiv": search_arxiv(queries),
            "huggingface": search_huggingface_datasets(queries)
        }

        context["survey_results"] = results
        return context
