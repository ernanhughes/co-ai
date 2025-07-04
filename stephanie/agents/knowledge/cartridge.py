from stephanie.agents.world.base_agent import BaseAgent
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import hashlib
from stephanie.models.cartridge import CartridgeORM

CARTRIDGE_UPDATE_THRESHOLD = 0.7  # Threshold for updating cartridges based on feedback

    
class CartridgeAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)


    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        cartridges = []
        cartridge_domains = set()

        for doc in documents:
            try:
                doc_id = doc["id"]
                title = doc.get("title", f"Document {doc_id}")
                summary = doc.get("summary", "")
                text = doc.get("content", doc.get("text", ""))
                goal_id = context.get("goal_id")  # optional

                full_text = f"# {title}\n\n## Summary\n\n{summary}\n\n## Content\n\n{text}"
                markdown_hash = hashlib.md5(full_text.encode()).hexdigest()
                filename = f"cartridge_{markdown_hash}.md"
                markdown_path = save_markdown_file(full_text, filename)

                # Get embedding
                embedding_vector = self.memory.embedding.get_or_create(text)
                embedding_vector_id = self.memory.embedding.get_id_from_text(text)


                # Get domain classification
                top_domains = self.classifier.classify(text)
                top_labels = [d[0] for d in top_domains]
                cartridge_domains.update(top_labels)

                # Create and store CartridgeORM
                cartridge = CartridgeORM(
                    goal_id=goal_id,
                    source_type="document",
                    source_uri=str(doc_id),
                    markdown_path=markdown_path,
                    embedding_id=embedding_entry.id,
                    created_at=datetime.utcnow(),
                )
                self.memory.session.add(cartridge)
                self.memory.session.flush()  # get cartridge.id

                cartridges.append({"id": cartridge.id, "domains": top_labels})

            except Exception as e:
                self.logger.log(
                    "DocumentLoadFailed", {"id": doc.get("id"), "error": str(e)}
                )

        context[self.output_key] = cartridges
        context["document_ids"] = [cartridge.get("id") for cartridge in cartridges]
        context["document_domains"] = cartridge_domains
        return context
