from co_ai.evaluator.base import BaseEvaluator
from co_ai.reasoning.arm.format_selector import AdaptiveReasoningSelector
from co_ai.reasoning.arm.utils import REASONING_FORMATS


class ARMReasoningEvaluator(BaseEvaluator):
    def __init__(self, memory, logger, device="cpu", mode="adaptive"):
        super().__init__(memory, logger, device)
        self.selector = AdaptiveReasoningSelector(
            model=None,  # Inject later
            tokenizer=None,
            memory=memory,
            logger=logger,
            device=device
        )
        self.mode = mode  # 'adaptive', 'instruction_guided', 'consensus'

    def judge(self, goal, prompt, output_a, output_b):
        # Reuse Mr Q's existing judgment mechanism
        return super().judge(goal, prompt, output_a, output_b)

    def train_from_database(self, goal: str, cfg: dict):
        # Override or extend as needed
        pass

    def retrieve_similar(self, prompt, k=3):
        return self.memory.prompt.get_similar(prompt, k)

    def generate(self, prompt: str) -> str:
        if self.mode == "adaptive":
            fmt = self.selector.select_format(prompt)
        elif self.mode == "instruction_guided":
            fmt = self._infer_instruction(prompt)
        elif self.mode == "consensus":
            return self._consensus_generate(prompt)
        else:
            fmt = "direct"

        return self.selector.generate_with_format(prompt, fmt)

    def _infer_instruction(self, prompt: str) -> str:
        for fmt, token in REASONING_FORMATS.items():
            if prompt.startswith(token):
                return fmt
        return "direct"

    def _consensus_generate(self, prompt: str) -> str:
        direct = self.selector.generate_with_format(prompt, "direct")
        short_cot = self.selector.generate_with_format(prompt, "short_cot")
        code = self.selector.generate_with_format(prompt, "code")

        answers = set([self._extract_answer(direct),
                       self._extract_answer(short_cot),
                       self._extract_answer(code)])

        if len(answers) == 1:
            return direct
        else:
            return self.selector.generate_with_format(prompt, "long_cot")