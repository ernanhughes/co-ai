
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType


class HNetAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        documents = context.get(self.input_key, [])
        results = []

        for doc in documents:
            doc_id = doc["id"]
            goal = context.get("goal", "")
            text = doc.get("summary") or doc.get("content", "")
            scorable = Scorable(text=text, target_type=TargetType.DOCUMENT, id=doc_id)
            print(f"Processing document {doc_id} text: {text[:50]}...")
            embedding = self.memory.hnet_embeddings.get_or_create(text)
            print(f"Embedding for hnet document {doc_id} created: {embedding[:10]}...")
            embedding = self.memory.hf_embeddings.get_or_create(text)
            print(f"Embedding for hf document {doc_id} created: {embedding[:10]}...")


            results.append(scorable.to_dict())

        context[self.output_key] = results
        return context

