from co_ai.agents.base import BaseAgent
from co_ai.constants import GOAL
from co_ai.tools.arxiv_tool import search_arxiv
from co_ai.tools.huggingface_tool import search_huggingface_datasets
from co_ai.tools import WebSearchTool  # Assume this exists
from co_ai.memory.search_result_store import SearchResultStore


class SearchOrchestratorAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.web_search_tool = WebSearchTool(cfg.get("web_search", {}), self.logger)
        self.max_results = cfg.get("max_results", 5)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        queries = context.get("search_queries", [])
        goal_id = goal.get("id")
        results = []

        for query in queries:
            search_query = query.get("goal_text")
            source = self.route_query(search_query)
            try:
                if source == "arxiv":
                    hits = await search_arxiv([search_query])
                elif source == "huggingface":
                    hits = await search_huggingface_datasets([search_query])
                elif source == "web":
                    hits = await self.web_search_tool.search(
                        search_query, max_results=self.max_results
                    )
                else:
                    continue

                enriched_hits = [
                    {
                        "query": search_query,
                        "source": source,
                        "result_type": hit.get("type", "unknown"),
                        "title": hit.get("title", hit.get("name", "")),
                        "summary": hit.get("snippet", hit.get("description", "")),
                        "url": hit.get("url", ""),
                        "goal_id": goal_id,
                        "parent_goal": goal.get("goal_text"),
                        "strategy": goal.get("strategy"),
                        "focus_area": goal.get("focus_area"),
                        "extra_data": {
                            "source_specific": hit
                        }
                    }
                    for hit in hits
                ]

                # Store results in DB
                stored_results = self.memory.search_results.bulk_add_results(enriched_hits)
                results.extend(stored_results)

            except Exception as e:
                self.logger.log(
                    "SearchToolFailed",
                    {"query": search_query, "tool": source, "error": str(e)}
                )

        # Save result IDs or ORM objects back to context
        context["search_result_ids"] = [r.id for r in results]
        context["search_results"] = [r.to_dict() for r in results]
        return context

    def route_query(self, query: str) -> str:
        """
        Decide which source to use based on query content.
        """
        query_lower = query.lower()
        if any(kw in query_lower for kw in ["paper", "study", "theory", "method"]):
            return "arxiv"
        elif any(kw in query_lower for kw in ["dataset", "model", "huggingface"]):
            return "huggingface"
        else:
            return "web"