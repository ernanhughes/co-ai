from co_ai.agents.base import BaseAgent
from co_ai.agents.mixins.prompt_evolver_mixin import PromptEvolverMixin
from co_ai.compiler.llm_compiler import LLMCompiler
from co_ai.compiler.passes.strategy_mutation_pass import StrategyMutationPass
from co_ai.constants import GOAL
import dspy
from co_ai.evaluator.evaluator_loader import get_evaluator

class PromptCompilerAgent(BaseAgent, PromptEvolverMixin):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.prompt_key = cfg.get("prompt_key", "default")
        self.sample_size = cfg.get("sample_size", 20)
        self.generate_count = cfg.get("generate_count", 10)
        self.version = cfg.get("version", 1)
        self.use_strategy_mutation = cfg.get("use_strategy_mutation", False)

        # Initialize the LLM compiler through DSPy
        llm = dspy.LM("ollama_chat/qwen3", api_base="http://localhost:11434")
        self.init_evolver(llm, logger=logger)
        self.compiler = LLMCompiler(llm=self.llm, logger=self.logger)
        self.evaluator = get_evaluator(cfg, memory, self.call_llm, logger)
        if self.use_strategy_mutation:
            self.strategy_pass = StrategyMutationPass(self.evaluator,
                compiler=self.compiler, logger=self.logger
            )


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

        # Score and sort prompts
        scored = []
        for prompt_text in refined_prompts:
            score = self.score_prompt(
                prompt=prompt_text,
                reference_output=examples[0].get("hypothesis_text", ""),
                context=context
            )
            scored.append((prompt_text, score))

        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        context["refined_prompts"] = scored_sorted

        # Log top results
        if self.logger:
            for i, (text, score) in enumerate(scored_sorted[:5]):
                self.logger.log(
                    "CompiledPromptScore",
                    {"rank": i + 1, "score": score, "prompt": text[:200]},
                )

        return context

    def score_prompt(self, prompt: str, reference_output, context:dict) -> float:
        if not self.evaluator:
            return 0.0
        try:
            return self.evaluator.score_single(prompt, reference_output, context)
        except Exception as e:
            if self.logger:
                self.logger.log("PromptScoreError", {"prompt": prompt[:100], "error": str(e)})
            return 0.0