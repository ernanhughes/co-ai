from dependency_injector.wiring import Provide, inject

from stephanie.agents.base_agent import BaseAgent
from stephanie.memory.hnet_embedding_store import HNetEmbeddingStore
from stephanie.protocols.embedding.base import EmbeddingProtocol
from stephanie.protocols.embeddings.embeddings import EmbeddingProtocol
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.score_bundle import ScoreBundle
from stephanie.tools.hnet_embedder import (PoolingStrategy,
                                           StephanieHNetChunker,
                                           StephanieHNetEmbedder)


class HnetAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None,  embedder: EmbeddingProtocol = Provide["container.embedder_selector"]):
        super().__init__(cfg, memory, logger)
        self.embedding_store = HNetEmbeddingStore(cfg, memory.conn, memory.db, logger=logger)
        embedder: EmbeddingProtocol = Provide["container.embedder_selector"]

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        for doc in documents:
            doc_id = doc["id"]
            goal = context.get("goal", "")
            text = doc.get("summary") or doc.get("content", "")
            scorable = Scorable(text=text, target_type=TargetType.DOCUMENT, id=doc_id)

            embedding = self.embedder.embed(text)
            context["embedding"] = embedding
            self.memory.hnet_embedding_store.save_embedding(text, embedding)



            results.append(scorable)

        context[self.output_key] = results
        return context

        text = context.get("input_text")
        if not text:
            return context

        try:
            embedding = self.embedder.embed(text)
            context["embedding"] = embedding
            self.memory.hnet_embedding_store.save_embedding(text, embedding)
        except Exception as e:
            context["embedding_error"] = str(e)
            self.logger.log("HNetEmbeddingFailed", {"error": str(e)})

        return context