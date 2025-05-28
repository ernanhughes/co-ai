import random
import torch
from co_ai.reasoning.arm.interface import ReasoningFormatSelector
from co_ai.reasoning.arm.utils import REASONING_FORMATS

class AdaptiveReasoningSelector(ReasoningFormatSelector):
    def __init__(self, *args, num_samples=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples

    def select_format(self, prompt: str) -> str:
        responses = []
        scores = []

        for _ in range(self.num_samples):
            fmt = random.choice(list(REASONING_FORMATS.keys()))
            response = self.generate_with_format(prompt, fmt)
            score = self._score_response(prompt, response)
            responses.append(response)
            scores.append(score)

        best_idx = scores.index(max(scores))
        best_response = responses[best_idx]
        best_fmt = self._detect_format(best_response)

        return best_fmt

    def generate_with_format(self, prompt: str, fmt: str) -> str:
        prefix = REASONING_FORMATS.get(fmt, "")
        full_prompt = f"{prefix}{prompt}"
        # Replace with actual generation call to self.model
        return self.model.generate(full_prompt)

    def _score_response(self, prompt: str, response: str) -> float:
        # Use value predictor + KL penalty or memory-based similarity
        prompt_emb = torch.tensor(self.memory.embedding.get_or_create(prompt)).unsqueeze(0).to(self.device)
        output_emb = torch.tensor(self.memory.embedding.get_or_create(response)).unsqueeze(0).to(self.device)

        zsa = self.encoder(prompt_emb, output_emb)
        score = self.value_predictor(zsa).item()
        token_len = len(response.split())
        return score - 0.01 * token_len  # Penalize longer outputs

    def _detect_format(self, text):
        for fmt, token in REASONING_FORMATS.items():
            if token in text:
                return fmt
        return "unknown"