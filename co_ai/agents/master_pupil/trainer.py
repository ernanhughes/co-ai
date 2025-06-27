import numpy as np
from co_ai.agents.master_pupil.finetuner import PupilFineTuner
from co_ai.agents.base_agent import BaseAgent
import torch

class TrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None, master=None, pupil=None):
        super().__init__(cfg, memory, logger)
        self.master = master
        self.pupil = pupil
        self.embedding_store = memory.embedding
        self.finetuner = PupilFineTuner(
            input_dim=1024,  # match embedding dim of pupil
            output_dim=1024  # match embedding dim of master
        )

    async def run(self, context: dict) -> dict:
        """
        This method is not used in the current implementation.
        The alignment is done through align_response method.
        """
        return context

    def align_response(self, question, context=None, epochs=3):
        master_answer = self.master.answer(question, context)
        self.logger.log("MasterAnswer", {"master_answer": master_answer})
        pupil_answer = self.pupil.answer(question, context)
        self.logger.log("PupilAnswer", {"pupil_answer": pupil_answer})

        master_emb = np.array(self.embedding_store.get_or_create(master_answer))
        pupil_emb = np.array(self.embedding_store.get_or_create(pupil_answer))

        print(f"Initial pupil answer:\n{pupil_answer}\n")

        for i in range(epochs):
            student_input = torch.tensor(pupil_emb, dtype=torch.float32)
            teacher_output = torch.tensor(master_emb, dtype=torch.float32)

            loss = self.finetuner.train_step(student_input, teacher_output)
            print(f"Epoch {i+1} Loss: {loss:.4f}")

        return pupil_answer  # or return aligned result if reverse decoder added

    def predict_embedding(self, text):
        emb = np.array(self.embedding_store.get_or_create(text))
        input_tensor = torch.tensor(emb, dtype=torch.float32)
        with torch.no_grad():
            aligned = self.finetuner.model(input_tensor).numpy()
        return aligned


    def _approximate_generation_from_embedding(self, emb):
        """
        Dummy function. In practice, this could be replaced by a reverse lookup
        in an embedding database or use a generator model.
        """
        return " ".join(["generated"] * 10)