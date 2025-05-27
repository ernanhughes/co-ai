# co_ai/agents/idea_evaluator.py
from typing import Dict, List, Optional
from co_ai.agents.base import BaseAgent
from co_ai.evaluator.llm_judge_evaluator import LLMJudgeEvaluator
from co_ai.evaluator.mrq_self_evaluator import MRQSelfEvaluator


class IdeaEvaluatorAgent(BaseAgent):
    """
    Evaluates research ideas and hypotheses using multiple strategies:
    
    - LLM-based pairwise comparison (like DPO)
    - Preference learning via MR.Q Self Evaluator
    
    Aligns with NOVELSEEK's:
    > "Assessment Agent evaluates ideas across key dimensions"
    > "Self-Evolving Agent evolves each idea into 3 variants and selects top ones"
    """

    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.strategy = cfg.get("strategy", "llm")  # llm | mrq
        self.evaluator = self._init_evaluator()
        self.top_k = cfg.get("top_k", 5)

    def _init_evaluator(self):
        if self.cfg.get("evaluator", "llm") == "llm":
            llm_model = self.cfg.get("evaluator_model", self.cfg.get("model"))
            prompt_file = self.cfg.get("evaluator_prompt_file", "prompts/mrq_scoring.j2")
            return LLMJudgeEvaluator(
                self.cfg,
                llm_cfg=self.cfg,
                prompt_file=prompt_file,
                llm=self.call_llm,
                logger=self.logger
            )
        else:
            return MRQSelfEvaluator(
                memory=self.memory,
                logger=self.logger,
                device=self.cfg.get("device", "cpu")
            )

    async def run(self, context: dict) -> dict:
        """
        Evaluate hypotheses and select top performers.
        
        Args:
            context (dict): Contains 'hypotheses' list and 'goal'
            
        Returns:
            dict: Updated context with scored hypotheses
        """
        hypotheses = context.get("hypotheses", [])
        goal = context.get("goal", {}).get("goal_text", "")
        baseline_hypothesis = context.get("baseline_hypothesis")

        if not hypotheses:
            self.logger.log("NoHypothesesToEvaluate", {})
            context["scored_hypotheses"] = []
            return context

        scored_results = []

        if self.strategy == "llm":
            scored_results = await self._evaluate_with_llm(hypotheses, goal, baseline_hypothesis)

        elif self.strategy == "mrq":
            scored_results = self._evaluate_with_mrq(hypotheses, goal, baseline_hypothesis)

        else:
            raise ValueError(f"Unknown evaluator strategy: {self.strategy}")

        # Sort by composite score
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        context["scored_hypotheses"] = scored_results
        context["top_hypothesis"] = scored_results[0]

        return context

    async def _evaluate_with_llm(self, hypotheses: List[str], goal: str, baseline: Optional[str]) -> List[Dict]:
        """Use LLMJudgeEvaluator to compare each hypothesis with baseline"""
        results = []
        for hyp in hypotheses:
            preferred, scores = self.evaluator.judge(goal=goal, prompt=hyp, output_a=baseline, output_b=hyp)
            results.append({
                "text": hyp,
                "preferred": preferred,
                "scores": scores,
                "source": "llm-judge",
                "score": scores.get("score_b", 0),
                "reasoning": scores.get("reason", "")
            })
        return results

    def _evaluate_with_mrq(self, hypotheses: List[Dict], goal: str, baseline: Optional[str]) -> List[Dict]:
        """Use MRQSelfEvaluator to score hypotheses using embedding-based judgment"""
        results = []
        for hyp in hypotheses:
            try:
                _, scores = self.evaluator.judge(goal, hyp["text"], baseline or hyp["text"])
                results.append({
                    "text": hyp["text"],
                    "scores": {
                        "value_a": scores["value_a"],
                        "value_b": scores["value_b"]
                    },
                    "source": "mrq",
                    "score": scores["value_b"],
                    "reasoning": "Based on learned preference model"
                })
            except Exception as e:
                self.logger.log("MRQEvaluationFailed", {"error": str(e)})
                continue
        return results

    def get_top_k(self, context:dict, k: int = 5):
        """Helper to retrieve top k hypotheses after evaluation"""
        return sorted(
            context["scored_hypotheses"],
            key=lambda x: x["score"],
            reverse=True
        )[:k]