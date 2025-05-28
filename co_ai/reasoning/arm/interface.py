from abc import ABC, abstractmethod


class ReasoningFormatSelector(ABC):
    def __init__(self, model, tokenizer, memory, logger, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.logger = logger
        self.device = device

    @abstractmethod
    def select_format(self, prompt: str) -> str:
        """Given a prompt, return the selected reasoning format tag."""
        pass

    @abstractmethod
    def generate_with_format(self, prompt: str, fmt: str) -> str:
        """Generate response using the given format."""
        pass