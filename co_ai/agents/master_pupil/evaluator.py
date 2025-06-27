
from co_ai.agents.base_agent import BaseAgent
from co_ai.agents.master_pupil.trainer import TrainerAgent
from torch.nn.functional import cosine_similarity
from co_ai.agents.master_pupil.master import MasterAgent
from co_ai.agents.master_pupil.pupil import PupilAgent  
import torch



class EvaluatorAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.master = MasterAgent(cfg, memory, logger)
        self.pupil = PupilAgent(cfg, memory, logger)
        self.trainer = TrainerAgent(cfg, memory, logger, master=self.master, pupil=self.pupil)

    async def run(self, context: dict) -> dict:
        question = context.get("goal", {}).get("goal_text", "")
        sim = self.trainer.align_response(question, context, epochs=5)
        context["similarity"] = sim
        self.logger.log("EvaluatorRun", {"similarity": sim})
        return context


    def score_alignment(self, text1, text2):
        emb1 = self.memory.embedding.get_or_create(text1)
        emb2 = self.memory.embedding.get_or_create(text2)
        sim = cosine_similarity([emb1], [emb2])[0][0]
        return sim

    def evaluate(self, question, master_answer, pupil_answer):
        score_before = self.score_alignment(master_answer, pupil_answer)
        aligned_answer = self.trainer.align_response(question)
        score_after = self.score_alignment(master_answer, aligned_answer)
        return {
            "before": score_before,
            "after": score_after,
            "improvement": score_after - score_before
        }
    

    def evaluate_alignment(self, master_output: torch.Tensor, pupil_output: torch.Tensor):
        similarity = cosine_similarity(master_output, pupil_output, dim=-1).mean().item()
        distance = torch.norm(master_output - pupil_output, dim=-1).mean().item()
        return {
            "cosine_similarity": round(similarity, 4),
            "vector_distance": round(distance, 4)
        }
