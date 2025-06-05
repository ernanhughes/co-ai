from co_ai.compiler.llm_compiler import LLMCompiler
from co_ai.compiler.prompt_evaluator import EvaluationResult
from co_ai.models.prompt_program import PromptProgramORM
import dspy
from dspy import BootstrapFewShot, Predict, Example, Signature, InputField, OutputField


class PromptTuningSignature(Signature):
    goal = InputField(desc="Scientific research goal or question")
    input_prompt = InputField(desc="Original prompt used to generate hypotheses")
    hypotheses = InputField(desc="Best hypothesis generated")
    review = InputField(desc="Expert review of the hypothesis")
    score = InputField(desc="Numeric score evaluating the hypothesis quality")
    refined_prompt = OutputField(desc="Improved version of the original prompt")


class PromptEvolver:
    def __init__(self, llm, logger=None):
        self.llm = llm
        self.logger = logger
        dspy.configure(lm=self.llm)

    def evolve(self, examples: list[dict], context: dict = {}, sample_size: int = 10) -> list[str]:
        """
        Use DSPy to tune prompts based on performance signals.
        Returns a list of refined prompt strings.
        """
        if not examples:
            return []

        training_set = [
            Example(
                goal=ex["goal"],
                input_prompt=ex["prompt_text"],
                hypotheses=ex["hypothesis_text"],
                review=ex.get("review", ""),
                score=ex.get("elo_rating", 1000),
            ).with_inputs("goal", "input_prompt", "hypotheses", "review", "score")
            for ex in examples[:sample_size]
        ]

        def fallback_metric(example, pred, trace=None):
            return 1.0  # fallback metric for training

        tuner = BootstrapFewShot(metric=fallback_metric)
        student = Predict(PromptTuningSignature)
        tuned_program = tuner.compile(student=student, trainset=training_set)

        refined_prompts = []
        for ex in examples[sample_size:]:
            try:
                result = tuned_program(
                    goal=ex["goal"],
                    input_prompt=ex["prompt_text"],
                    hypotheses=ex["hypothesis_text"],
                    review=ex.get("review", ""),
                    score=ex.get("elo_rating", 1000),
                )
                refined = result.refined_prompt.strip()
                refined_prompts.append(refined)
            except Exception as e:
                if self.logger:
                    self.logger.log("DSPyPromptEvolutionFailed", {"error": str(e)})

        return refined_prompts
