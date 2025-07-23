# stephanie/agents/learning/gild_trainer.py
import torch
import torch.nn.functional as F
from stephanie.agents.base_agent import BaseAgent

from stephanie.agents.inference.mrq_inference import MRQInferenceAgent
from stephanie.constants import GOAL

from stephanie.models.evaluation import EvaluationORM
from stephanie.scoring.scorable_factory import TargetType, ScorableFactory
from stephanie.scoring.scorable import Scorable
from stephanie.agents.inference.ebt_inference import EBTInferenceAgent
from stephanie.agents.inference.llm_inference import LLMInferenceAgent



class GILDTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.use_gild = cfg.get("use_gild", False)
        self.il_weight = cfg.get("il_weight", 0.5)
        self.il_decay_rate = cfg.get("il_decay_rate", 0.95)
        self.uncertainty_threshold = cfg.get("uncertainty_threshold", 0.3)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger.log(
            "GILDTrainerAgentInitialized",
            {
                "use_gild": self.use_gild,
                "il_weight": self.il_weight,
                "il_decay_rate": self.il_decay_rate,
                "dimensions": cfg.get("dimensions", ["alignment", "clarity"]),
            },
        )
        self.ebt = EBTInferenceAgent(cfg.get("ebt"), self.memory, self.logger)
        self.llm = LLMInferenceAgent(cfg.get("llm"), self.memory, self.logger)
        self.mrq = MRQInferenceAgent(cfg.get("mrq"), self.memory, self.logger)
        self.sicql = MRQInferenceAgent(cfg.get("sicql"), self.memory, self.logger)



    async def run(self, context: dict) -> dict:
        """
        Main GILD training loop
        1. Get demonstrations (EBT, LLM, or previous runs)
        2. Train with GILD objective
        3. Update belief cartridges with new policy
        """
        goal = context.get(GOAL)
        documents = context.get("documents", [])
        dimension = context.get("dimension", "alignment")

        # Track policy improvement
        policy_improvement = {
            "dimension": dimension,
            "improvement_steps": [],
            "policy_drift": [],
        }

        for doc in documents:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            expert_policy = self._get_expert_policy(context, scorable, dimension)

            # Get current policy from context
            current_policy = context.get("policy", {})
            if not current_policy:
                current_policy = self._default_policy(scorable, dimension)

            # Meta-train GILD objective
            if self.use_gild and expert_policy:
                gild_loss = self._compute_gild_loss(
                    current_policy, expert_policy
                )
                policy_improvement["improvement_steps"].append(
                    {"document_id": scorable.id, "gild_loss": gild_loss}
                )

                # Update weights dynamically
                self._update_weights(gild_loss)

                # Log to database with extra_data
                self._log_policy_improvement(
                    context,
                    scorable.id,
                    dimension,
                    current_policy,
                    expert_policy,
                    gild_loss,
                )

            # Update context with new policy
            improved_policy = self._apply_gild_update(
                current_policy, expert_policy
            )
            context["policy"] = improved_policy

        context["policy_improvement"] = policy_improvement
        return context

    def _get_expert_policy(self, context, scorable, dimension):
        """Get expert demonstration for GILD"""
        goal_text = context.get(GOAL, {}).get("goal_text", "")
        if self.cfg.get("use_ebt_as_expert", True):
            return self.ebt.get_refinements(goal_text, scorable.text, dimension)
        if self.cfg.get("use_llm_as_expert", True):
            return self.memory.llm_scorer.get_expert_policy(
                scorable.text, dimension
            )

    def _compute_gild_loss(self, current_policy, expert_policy):
        """Compute GILD loss between current and expert policy"""
        if isinstance(current_policy, dict):
            current_policy = torch.tensor(list(current_policy.values()))
        if isinstance(expert_policy, dict):
            expert_policy = torch.tensor(list(expert_policy.values()))

        # KL divergence for imitation loss
        return F.kl_div(
            F.log_softmax(current_policy, dim=-1),
            F.softmax(expert_policy, dim=-1),
            reduction="batchmean",
        )

    def _apply_gild_update(self, current_policy, expert_policy):
        """Apply GILD update to policy"""
        if not self.use_gild:
            return current_policy

        # Blend current and expert policy
        return {
            k: current_policy[k] * (1 - self.il_weight)
            + expert_policy[k] * self.il_weight
            for k in current_policy
        }

    def _update_weights(self, loss):
        """Decay imitation weight over time"""
        if loss < self.cfg.get("gild_loss_threshold", 0.1):
            self.il_weight *= self.il_decay_rate
            self.il_weight = max(
                self.il_weight, self.cfg.get("min_weight", 0.01)
            )

    def _log_policy_improvement(
        self, context, doc_id, dimension, policy, expert, loss
    ):
        """Log policy improvement to database"""
        evaluation = (
            self.memory.session.query(EvaluationORM)
            .filter_by(target_id=doc_id, dimension=dimension)
            .first()
        )

        if not evaluation:
            evaluation = EvaluationORM(
                goal_id=context.get("goal", {}).get("goal_id"),
                target_id=doc_id,
                target_type="document",
                evaluator_name="gild_trainer",
                model_name=f"gild_{dimension}",
                extra_data={},
            )
            self.memory.session.add(evaluation)

        # Update extra_data with GILD metrics
        evaluation.extra_data.update(
            {
                "gild_loss": float(loss.item()),
                "policy_weights": {
                    "il_weight": self.il_weight,
                    "rl_weight": 1 - self.il_weight,
                },
                "policy": policy,
                "expert_policy": expert,
            }
        )

        self.memory.session.commit()

    def _default_policy(
        self, scorable: Scorable, dimension: str
    ) -> dict[str, float]:
        """
        Generates a default policy when no prior policy exists.
        Returns policy logits in a structure compatible with GILD training.
        """
        # Option 1: Use dimension-specific prior (if available)
        if dimension in self.cfg.get("dimension_priors", {}):
            return {"policy_logits": self.cfg["dimension_priors"][dimension]}

        # Option 2: Use global prior from config
        global_prior = self.cfg.get("global_prior", 0.5)
        if isinstance(global_prior, (float, int)):
            # Single-action policy (shape: [0.5])
            return {"policy_logits": [global_prior]}

        # Option 3: Use uniform distribution for multi-action policy
        action_dim = self.cfg.get("action_dim", 1)
        if action_dim > 1:
            # Multi-action policy (shape: [0.3, 0.7])
            return {"policy_logits": [1.0 / action_dim] * action_dim}

        # Final fallback: Use average of past scores
        avg_score = self.memory.scores.get_average_score(dimension)
        return {
            "policy_logits": [avg_score or 0.5],
            "source": "historical_average",
        }
