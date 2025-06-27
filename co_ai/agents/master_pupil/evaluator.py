
from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL



class EvaluatorAgent:
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)

    def score_alignment(self, text1, text2):
        emb1 = self.memory.embedding.get_or_create(text1)
        emb2 = self.memory.embedding.get_or_create(text2)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return sim

    def evaluate(self, question, master_answer, pupil_answer):
        score_before = self.score_alignment(master_answer, pupil_answer)
        aligned_answer = trainer.align_response(question)
        score_after = self.score_alignment(master_answer, aligned_answer)
        return {
            "before": score_before,
            "after": score_after,
            "improvement": score_after - score_before
        }