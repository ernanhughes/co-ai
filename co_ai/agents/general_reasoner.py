from co_ai.agents.base import BaseAgent
from co_ai.evaluator import LLMJudgeEvaluator
from co_ai.evaluator import MRQSelfEvaluator
from co_ai.constants import GOAL, GOAL_TYPE  
from co_ai.models import Hypothesis
from co_ai.prompts import PromptLoader


class GeneralReasonerAgent(BaseAgent):
    def __init__(self, cfg, memory, logger):
        super().__init__(cfg, memory, logger)
        self.logger.log("AgentInit", {"agent": "GeneralReasonerAgent"})
        self.judge = self._init_judge()
        print(self.cfg)
        self.prompt_loader = PromptLoader(self.cfg, self.logger)

    async def run(self, context: dict):
        goal = context.get(GOAL)
        self.logger.log("AgentRunStarted", {"goal": goal})

        # Generate multiple reasoning outputs
        hypotheses = self.generate_hypotheses(goal, context)

        # Evaluate each hypothesis
        evaluations = []
        for i, hyp in enumerate(hypotheses):
            eval_result = self.judge.evaluate(goal, hyp)
            evaluations.append(eval_result)

            # Log everything for audit and analysis
            self.logger.log(
                {
                    "event": "EvaluationComplete",
                    "goal_id": goal["id"],
                    "hypothesis_id": hyp.id,
                    "hypothesis_text": hyp.text,
                    "strategy": hyp.features.get("strategy"),
                    "score": eval_result.get("score"),
                    "rationale": eval_result.get("reason", ""),
                    "evaluator": self.cfg.get("judge", "mrq"),
                }
            )

        # Return the best hypothesis
        best_idx = max(
            range(len(evaluations)), key=lambda j: evaluations[j].get("score", 0)
        )
        best_hypothesis = hypotheses[best_idx]
        best_hypothesis.evaluation = evaluations[best_idx]  # optional for later use

        return best_hypothesis

    def generate_hypotheses(self, question, context):
        # Simple loop; replace with model call w/ temperature or variations
        strategies = self.cfg.get("generation_strategy_list", ["cot"])
        merged = {**context, **{"question": question}}
        hypotheses = []
        for strategy in strategies:
            prompt = self.prompt_loader.from_file(f"strategy_{strategy}.txt", self.cfg, merged)
            response = self.call_llm(prompt, merged)
            hypothesis = Hypothesis(text=response, goal=context.get(GOAL), goal_type=context.get(GOAL_TYPE), 
                                    strategy_used=strategy, features={"strategy": strategy},
                                    source=self.name)
            self.memory.hypotheses.store(hypothesis)
            hypotheses.append(hypothesis)
        return hypotheses


    def _init_judge(self):
        if self.cfg.get("judge", "mrq") == "llm":
            llm = self.cfg.get("judge_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("judge_prompt_file", "judge_pairwise_comparison.txt")
            self.logger.log("EvaluatorInit", {"strategy": "LLM", "prompt_file": prompt_file})
            return LLMJudgeEvaluator(self.cfg, llm, prompt_file, self.call_llm, self.logger)
        else:
            self.logger.log("EvaluatorInit", {"strategy": "MRQ"})
            return MRQSelfEvaluator(self.memory, self.logger)
