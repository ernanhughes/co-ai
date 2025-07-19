# stephanie/agents/knowledge/knowledge_db_loader.py

from stephanie.agents.base_agent import BaseAgent
from stephanie.constants import GOAL


class KnowledgeDBLoaderAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.top_k = cfg.get("top_k", 100)
        self.include_full_text = cfg.get("include_full_text", False)
        self.search_method = cfg.get(
            "search_method", "document"
        )  # or "section"

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)
        goal_text = goal.get("goal_text", "")

        docs = self.memory.ollama_embeddings.search_related_documents(
            goal_text, self.top_k
        )

        context[self.output_key] = docs
        context["retrieved_ids"] = [d["id"] for d in docs]
        self.logger.log(
            "KnowledgeDBLoaded",
            {"count": len(docs), "search_method": self.search_method},
        )

        return context
