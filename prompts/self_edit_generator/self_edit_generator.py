# co_ai/agents/seal/self_edit_generator_agent.py

from co_ai.agents.base_agent import BaseAgent
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SelfEditGeneratorConfig:
    prompt_file: str = "implication.j2"
    num_edits: int = 5
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 512


class SelfEditGeneratorAgent(BaseAgent):
    def __init__(
        self,
        cfg,
        memory=None,
        logger=None,
        config: SelfEditGeneratorConfig = SelfEditGeneratorConfig(),
    ):
        super().__init__(cfg, memory, logger)
        self.config = config
        self.prompt_files = self.cfg.get("prompt_files", [])

    async def run(self, context: dict) -> dict:
        all_edits = []
        for prompt_file in self.prompt_files:
            prompt = self.prompt_loader.from_file(
                prompt_file, config=self.cfg, context=context
            )
            self.logger.log("PromptFileLoaded", {"file": prompt_file})
            response = self.call_llm(prompt, context)
            print(f"Generated response: {response}...")
            strategy = prompt_file.replace(".txt", "")
            all_edits.append({"edit": response, "strategy": strategy})

        context.setdefault("self_edits", []).append({"edit", response})
        return context
