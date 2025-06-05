from co_ai.agents.base import BaseAgent
from co_ai.agents.mixins.prompt_evolver_mixin import PromptEvolverMixin
from co_ai.constants import GOAL
import dspy


class PromptCompilerAgent(BaseAgent, PromptEvolverMixin):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.prompt_key = cfg.get("prompt_key", "default")
        self.sample_size = cfg.get("sample_size", 200)
        self.generate_count = cfg.get("generate_count", 10)
        self.version = cfg.get("version", 1)

        # Initialize the LLM compiler through DSPy
        llm = dspy.LM("ollama_chat/qwen3", api_base="http://localhost:11434")
        self.init_evolver(llm, logger=logger)

    async def run(self, context: dict) -> dict:
        goal = self.extract_goal_text(context.get(GOAL))
        total_count = self.sample_size + self.generate_count

        examples = self.memory.prompt.get_prompt_training_set(goal, total_count)
        if not examples:
            self.logger.log("PromptCompilerSkipped", {"reason": "no_examples", "goal": goal})
            return context

        refined_prompts = self.evolve_prompts(
            examples, context=context, sample_size=self.sample_size
        )

        for prompt in refined_prompts:
            self.memory.prompt.save(
                goal={"goal_text": goal},
                agent_name=self.name,
                prompt_key=self.prompt_key,
                prompt_text=prompt,
                strategy="dspy_compilation",
                version=self.version + 1,
                pipeline_run_id=context.get("pipeline_run_id"),
            )
            self.add_to_prompt_history(context, prompt, {"source": "dspy_compiler"})

        self.logger.log(
            "PromptCompilerCompleted",
            {"goal": goal, "generated_count": len(refined_prompts)},
        )
        return context
 