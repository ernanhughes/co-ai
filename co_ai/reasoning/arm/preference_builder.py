from typing import Tuple, List
from collections import Counter
from co_ai.reasoning.arm.format_selector import ReasoningFormatSelector
from co_ai.reasoning.arm.utils import REASONING_FORMATS


class ARMPairwisePreferenceBuilder:
    def __init__(self, selector: ReasoningFormatSelector):
        self.selector = selector

    def build_preferences(self, prompt: str, ground_truth: str) -> List[Tuple[str, str]]:
        responses = [
            self.selector.generate_with_format(prompt, fmt)
            for fmt in REASONING_FORMATS.keys()
            for _ in range(2)
        ]
        rewards = self._compute_rewards(responses, ground_truth)
        weights = self._format_diversity_weight(responses, rewards)

        preferences = []
        for i in range(len(responses)):
            for j in range(len(responses)):
                if rewards[i] > rewards[j]:
                    preferences.append((responses[i], responses[j]))

        return preferences

    def _compute_rewards(self, responses, ground_truth):
        return [1 if self._extract_answer(r) == ground_truth else 0 for r in responses]

    def _extract_answer(self, response):
        if "<ANSWER>" in response:
            return response.split("<ANSWER>")[1].split("</ANSWER>")[0].strip()
        elif "The answer is" in response:
            return response.split("The answer is")[1].strip().rstrip(".")
        else:
            return response.strip()

    def _format_diversity_weight(self, responses, rewards):
        formats = [self.selector._detect_format(r) for r in responses]
        freq = Counter(formats)
        weights = []
        for i in range(len(responses)):
            f = formats[i]
            scale = (len(responses) / freq[f]) * (1 + (i % 2))  # simple decay
            weights.append(scale * rewards[i])
        return weights