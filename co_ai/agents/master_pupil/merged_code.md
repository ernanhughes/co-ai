<!-- Merged Python Code Files -->


## File: evaluator.py

`python

from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL

class EvaluatorAgent:
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    async def run(self, context: dict) -> dict:
        goal = context.get(GOAL)

    def score_alignment(self, text1, text2):
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
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
``n

## File: lord_attack.py

`python
import logging
from typing import Any, Optional

class LordAttackModule:
    def __init__(
        self,
        cfg: dict[str, Any],
        memory: callable,
        logger: Optional[logging.Logger] = None
    ):
        """
        A modular class for implementing LoRD-based model extraction attacks.

        Args:
            cfg (Dict[str, Any]): Configuration dictionary.
            memory (callable): Function that returns a store object (e.g., embedding_store).
            logger (Optional[logging.Logger]): Logger object for event logging.
        """
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        # Initialize memory/store objects using the provided memory function
        try:
            self.embedding_store = memory("embedding")
            self.prompt_store = memory("prompt")
            self.hypothesis_store = memory("hypothesis")
            self.logger.info("✅ Memory stores initialized.")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize memory stores: {e}")
            raise

        # Load config parameters
        self.victim_model = cfg.get("lord", {}).get("victim_model", "qwen3")
        self.student_model = cfg.get("lord", {}).get("student_model", "phi3")
        self.embedding_model = cfg.get("embeddings", {}).get("model", "mxbai-embed-large")
        self.temperature = cfg.get("lord", {}).get("temperature", 0.7)
        self.max_tokens = cfg.get("lord", {}).get("max_tokens", 256)
        self.log_frequency = cfg.get("lord", {}).get("log_frequency", 1)

        self.logger.debug("Intialized LordAttackModule with config:")
        self.logger.debug(self.cfg)

    def log_event(self, event_type: str, message: str, level: str = "info"):
        """
        Helper to log events with structured metadata.
        """
        log_data = {
            "event": event_type,
            "module": self.__class__.__name__,
            "message": message
        }
        if self.logger:
            if level == "debug":
                self.logger.debug(log_data)
            elif level == "warning":
                self.logger.warning(log_data)
            elif level == "error":
                self.logger.error(log_data)
            else:
                self.logger.info(log_data)
        else:
            print(f"[{event_type}] {message}")

    def query_ollama(self, model_name: str, prompt: str) -> str:
        """
        Query Ollama API for model response.
        """
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        try:
            import requests
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json().get("response", "")
            self.log_event("OllamaQuerySuccess", f"Queried {model_name} successfully.")
            return result
        except Exception as e:
            self.log_event("OllamaQueryFailed", f"Failed to query {model_name}: {str(e)}", level="error")
            return ""

    def get_embedding(self, text: str) -> list:
        """
        Get embedding via Ollama embedding model.
        """
        try:
            return self.embedding_store.get_or_create(text)
        except Exception as e:
            self.log_event("EmbeddingFailed", f"Failed to embed text: {str(e)}", level="error")
            return []

    def train_step(self, prompt: str, step: int):
        """
        Perform one training step of LoRD algorithm.
        """
        self.log_event("TrainingStepStart", f"Starting step {step}")

        # Step 1: Query victim model
        victim_response = self.query_ollama(self.victim_model, prompt)

        # Step 2: Query student model
        student_response = self.query_ollama(self.student_model, prompt)

        # Step 3: Embed responses
        victim_emb = self.get_embedding(victim_response)
        student_emb = self.get_embedding(student_response)

        # Step 4: Simulate loss calculation (approximate LoRD-style loss)
        # In real use, this would involve contrastive loss and KL divergence
        similarity_score = self._compute_similarity(victim_emb, student_emb)
        loss = 1.0 - similarity_score  # Dummy loss for illustration

        # Step 5: Log results
        if step % self.log_frequency == 0:
            self.log_event("TrainingStepResult", f"Step {step}, Loss: {loss:.4f}, Similarity: {similarity_score:.4f}")

        return loss

    def _compute_similarity(self, vec1: list, vec2: list) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

    def run_attack(self, prompt_base: str = "Explain quantum computing.", epochs: int = 10):
        """
        Run full LoRD attack loop.
        """
        self.log_event("AttackStarted", f"Running LoRD attack for {epochs} epochs")

        for step in range(epochs):
            full_prompt = f"{prompt_base} (Iteration {step})"
            loss = self.train_step(full_prompt, step)

            # Optional: Save prompt and result to memory
            self.prompt_store.save_prompt({
                "prompt": full_prompt,
                "loss": loss,
                "step": step
            })

        self.log_event("AttackFinished", "LoRD attack completed.")
``n

## File: master.py

`python
from co_ai.agents.base_agent import BaseAgent
from co_ai.utils.timing import time_function

class MasterAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    @time_function(logger=None)
    async def run(self, context: dict) -> dict:
        self.execute_prompt(context)
        return context
``n

## File: pupil.py

`python
from co_ai.agents.base_agent import BaseAgent
from co_ai.utils.timing import time_function

class PupilAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)

    @time_function(logger=None)
    async def run(self, context: dict) -> dict:
        self.execute_prompt(context)
        return context
        
``n

## File: student_finetuner.py

`python
# co_ai/agents/master_pupil/student_finetuner.py

import torch
import torch.nn as nn
import torch.optim as optim

class StudentModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class StudentFineTuner:
    def __init__(self, input_dim=512, output_dim=768, lr=1e-4):
        self.model = StudentModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, student_input, teacher_output):
        self.model.train()
        pred = self.model(student_input)
        loss = self.loss_fn(pred, teacher_output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
``n

## File: trainer.py

`python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from co_ai.agents.base_agent import BaseAgent
from co_ai.constants import GOAL

class TrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None, master=None, pupil=None):
        super().__init__(cfg, memory, logger)
        self.master = master
        self.pupil = pupil

    def align_response(self, question, epochs=3, lr=0.1):
        target_answer = self.master.answer(question)
        current_answer = self.pupil.answer(question)

        target_emb = self.memory.embedding.get_or_create(target_answer)
        current_emb = self.memory.embedding.get_or_create(current_answer)

        print(f"\nInitial Answer: {current_answer}")

        for i in range(epochs):
            # Simulate gradient update using embedding space
            diff = target_emb - current_emb
            updated_emb = current_emb + lr * diff

            # Approximate generation from updated embedding (via nearest neighbor?)
            # This part would ideally use vector-to-text decoding if available
            updated_text = self._approximate_generation_from_embedding(updated_emb)

            current_emb = updated_emb
            print(f"Epoch {i+1}: {updated_text}")

        return updated_text

    def _approximate_generation_from_embedding(self, emb):
        """
        Dummy function. In practice, this could be replaced by a reverse lookup
        in an embedding database or use a generator model.
        """
        return " ".join(["generated"] * 10)
``n
