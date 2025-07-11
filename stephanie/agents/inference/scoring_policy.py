from stephanie.agents.base_agent import BaseAgent
from stephanie.agents.inference.ebt_inference import EBTInferenceAgent
from stephanie.agents.inference.mrq_inference import MRQInferenceAgent
from stephanie.agents.inference.llm_inference import LLMInferenceAgent
from stephanie.scoring.scorable_factory import TargetType
from stephanie.scoring.scorable_factory import ScorableFactory
from stephanie.scoring.ebt.buffer import EBTTrainingBuffer
from sqlalchemy import text
from stephanie.scoring.ebt.refinement_trainer import EBTRefinementTrainer

import torch

DEFAULT_DIMENSIONS = [
    "alignment",
    "implementability",
    "clarity",
    "relevance",
    "novelty",
]


class ScoringPolicyAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.cfg = cfg
        self.memory = memory
        self.logger = logger

        self.dimensions = cfg.get("dimensions", DEFAULT_DIMENSIONS)

        self.ebt_refine_threshold = cfg.get("ebt_refinement_threshold", 0.7)
        self.llm_fallback_threshold = cfg.get("llm_fallback_threshold", 0.9)
        self.steps = cfg.get("optimization_steps", 10)
        self.step_size = cfg.get("step_size", 0.05)

        self.ebt = EBTInferenceAgent(self.cfg, memory, logger)
        self.mrq = MRQInferenceAgent(self.cfg, memory, logger)
        self.llm = LLMInferenceAgent(self.cfg, memory, logger)

        self.training_buffer_path = cfg.get(
            "training_buffer_path", "ebt_buffer.json"
        )
        self.training_buffer = EBTTrainingBuffer(
            self.logger, self.training_buffer_path
        )

    async def run(self, context: dict) -> dict:
        goal_text = context["goal"]["goal_text"]
        docs = context[self.input_key]
        results = []
        event_ids = []
        self.ebt.load_models(self.dimensions)
        self.mrq.load_models(self.dimensions)

        for doc in docs:
            scorable = ScorableFactory.from_dict(doc, TargetType.DOCUMENT)
            # 1. Initial MRQ score
            mrq_scores = self.mrq.score(goal_text, scorable.text)
            self.logger.log(
                "MRQScoresCalculated",
                {
                    "scorable": scorable.id,
                    "scores": mrq_scores,
                },
            )

            # Step 2: Estimate uncertainty using EBT energy
            ebt_energy = self.ebt.get_energy(goal_text, scorable.text)
            self.logger.log(
                "EBTEnergyCalculated",
                {"scorable": scorable.id, "energy": ebt_energy},
            )
            uncertainty_by_dim = {
                dim: torch.sigmoid(torch.tensor(raw)).item()
                for dim, raw in ebt_energy.items()
            }

            self.logger.log(
                "UncertaintyEstimated",
                {
                    "scorable": scorable.id,
                    "uncertainty_by_dimension": uncertainty_by_dim,
                },
            )

            refined = False
            refined_text = None
            # Step 3: Optional refinement (if any dimension exceeds EBT refine threshold)
            if any(
                u > self.ebt_refine_threshold
                for u in uncertainty_by_dim.values()
            ):
                refined = True
                refined_result = self.ebt.optimize(goal_text, scorable.text)
                refined_text = refined_result.get("refined_text")
                mrq_scores = self.mrq.score(goal_text, refined_text)
                self.logger.log(
                    "DocumentRefinedWithEBT", {"document_id": scorable.id}
                )
                refined_score = refined_result.get("final_energy")
                # Log disagreement for retraining
                self.training_buffer.maybe_add(
                    context=goal_text,
                    candidate=scorable.text,
                    llm_score=refined_score,
                    ebt_score=mrq_scores["alignment"],
                    threshold=self.cfg.get("disagreement_threshold", 0.15),
                    metadata={"dimension": "alignment"},
                )

            # Step 4: Optional LLM fallback (if any dimension exceeds fallback threshold)
            if any(
                u > self.llm_fallback_threshold
                for u in uncertainty_by_dim.values()
            ):
                llm_scores = self.llm.score(goal_text, doc)
                final_scores = llm_scores
                source = "llm"
                self.logger.log(
                    "LLMFallbackUsed", {"document_id": scorable.id}
                )
            else:
                final_scores = mrq_scores
                source = "mrq"

            # Log raw data for analysis
            result_entry = {
                "document_id": scorable.id,
                "original_text": scorable.text,
                "refined_text": refined_text if refined else scorable.text,
                "mrq_scores": mrq_scores,
                "ebt_energy": ebt_energy,
                "uncertainty_by_dimension": uncertainty_by_dim,
                "used_refinement": any(
                    u > self.ebt_refine_threshold
                    for u in uncertainty_by_dim.values()
                ),
                "used_llm_fallback": any(
                    u > self.llm_fallback_threshold
                    for u in uncertainty_by_dim.values()
                ),
                "final_source": source,
                # "steps_used": len(refinement_trace) if refined else 0,
                # "converged": refinement_trace[-1] - refinement_trace[0] < 0.05 if refined else None
            }

            # Log to database    # Inside run() after processing a document
            event_id = self._log_to_database(result_entry)
            event_ids.append(event_id)
            self.logger.log("ScoringEvent", result_entry)
            results.append(result_entry)

            self.logger.log(
                "ScoringPolicyCompleted",
                {
                    "document_id": scorable.id,
                    "final_scores": final_scores,
                    "source": source,
                },
            )

        context[self.output_key] = results
        context["event_ids"] = event_ids

        self.plot_score_distributions(results)
        self.generate_summary_report(results)
        return context

    def calculate_agreement(self, results):
        """Compare MRQ and LLM scores where both used"""
        import pandas as pd

        llm_mrq = []
        for result in results:
            if result["used_llm_fallback"]:
                for dim in self.dimensions:
                    llm_mrq.append(
                        {
                            "dimension": dim,
                            "mrq_score": result["mrq_scores"][dim],
                            "llm_score": result["scores"][dim],
                        }
                    )
        return pd.DataFrame(llm_mrq)

    # Helper method
    def _log_to_database(self, entry):
        # Insert into scoring_events table
        insert_event_sql = """
        INSERT INTO scoring_events (
            document_id,
            goal_text,
            original_text,
            refined_text,
            final_source,
            used_refinement,
            refinement_steps,
            used_llm_fallback
        )
        VALUES (
            :document_id,
            :goal_text,
            :original_text,
            :refined_text,
            :final_source,
            :used_refinement,
            :refinement_steps,
            :used_llm_fallback
        )
        RETURNING id
        """

        event_params = {
            "document_id": entry["document_id"],
            "goal_text": entry.get("goal_text", "UNKNOWN"),
            "original_text": entry.get("original_text"),
            "refined_text": entry.get("refined_text"),
            "final_source": entry["final_source"],
            "used_refinement": entry["used_refinement"],
            "refinement_steps": entry.get("steps_used", 0),
            "used_llm_fallback": entry["used_llm_fallback"],
        }

        event_id = (
            self.memory.session.execute(text(insert_event_sql), event_params)
            .fetchone()
            .id
        )

        # Insert into scoring_dimensions table
        insert_dim_sql = """
        INSERT INTO scoring_dimensions (
            event_id,
            dimension,
            mrq_score,
            ebt_energy,
            uncertainty_score,
            final_score
        )
        VALUES (
            :event_id,
            :dimension,
            :mrq_score,
            :ebt_energy,
            :uncertainty_score,
            :final_score
        )
        """

        for dim in self.dimensions:
            dim_params = {
                "event_id": event_id,
                "dimension": dim,
                "mrq_score": entry["mrq_scores"].get(dim),
                "ebt_energy": entry["ebt_energy"].get(dim),
                "uncertainty_score": entry["uncertainty_by_dimension"].get(
                    dim
                ),
                "final_score": entry["mrq_scores"].get(dim),
            }
            self.memory.session.execute(text(insert_dim_sql), dim_params)

        return event_id

    # In scoring_policy.py
    def plot_uncertainty_map(self, uncertainties, doc_id):
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        sns.heatmap(
            [uncertainties],  # single row for this document
            annot=True,
            cmap="YlOrRd",
            yticklabels=[doc_id],
            xticklabels=self.dimensions,
        )
        plt.title("Uncertainty Across Dimensions")
        plt.tight_layout()
        plt.savefig(f"uncertainty_maps/{doc_id}.png")
        plt.close()

        # After processing all documents

    def plot_score_distributions(self, results):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        df = pd.DataFrame(
            [
                {
                    "dimension": dim,
                    "source": result["final_source"],
                    "score": result["mrq_scores"][dim],
                }
                for result in results
                for dim in self.dimensions
            ]
        )

        plt.figure(figsize=(12, 6))
        sns.violinplot(x="dimension", y="score", hue="source", data=df)
        plt.xticks(rotation=45)
        plt.title("Score Distributions by Dimension and Source")
        plt.savefig("score_distributions.png")
        plt.close()

    def generate_summary_report(self, results):
        import json

        total = len(results)
        refined_count = sum(1 for r in results if r["used_refinement"])
        llm_fallback_count = sum(1 for r in results if r["used_llm_fallback"])

        summary = {
            "total_documents": total,
            "refined_documents": refined_count,
            "llm_fallback_rate": llm_fallback_count / total,
            "average_uncertainty": {
                dim: sum(r["uncertainty_by_dimension"][dim] for r in results)
                / total
                for dim in self.dimensions
            },
            "dimension_refinement_rate": {
                dim: sum(
                    1
                    for r in results
                    if r["uncertainty_by_dimension"][dim]
                    > self.ebt_refine_threshold
                )
                / total
                for dim in self.dimensions
            },
        }

        # Save to file
        with open("policy_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def analyze_refinement_impact(self, results):
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        impacts = []
        for result in results:
            if result["used_refinement"]:
                for dim in self.dimensions:
                    impact = result["scores"][dim] - result["mrq_scores"][dim]
                    impacts.append(
                        {
                            "dimension": dim,
                            "impact": impact,
                            "before": result["mrq_scores"][dim],
                            "after": result["scores"][dim],
                        }
                    )

        # Calculate average impact
        impacts_df = pd.DataFrame(impacts)
        avg_impact = impacts_df.groupby("dimension")["impact"].mean()

        print("Average Score Improvement from Refinement:")
        print(avg_impact)

        # Plot improvement distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="dimension", y="impact", data=impacts_df)
        plt.title("Refinement Impact by Dimension")
        plt.xticks(rotation=45)
        plt.savefig("refinement_impact.png")
        plt.close()

    def export_results(self, results):
        """Export results to CSV for external analysis"""
        import pandas as pd

        rows = []
        for result in results:
            base_row = {
                "document_id": result["document_id"],
                "used_refinement": result["used_refinement"],
                "used_llm_fallback": result["used_llm_fallback"],
            }

            # Add per-dimension data
            for dim in self.dimensions:
                base_row[f"{dim}_uncertainty"] = result[
                    "uncertainty_by_dimension"
                ][dim]
                base_row[f"{dim}_score"] = result["scores"][dim]
                base_row[f"{dim}_energy"] = result["ebt_energy"][dim]

            rows.append(base_row)

        # Save to CSV
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_csv("scoring_results.csv", index=False)
        return df

    # In your self-tuning agent
    def update_ebt(self):
        """Periodically update EBT models from refinement history"""
        examples = self._fetch_recent_refinements()
        if examples:
            trainer = EBTRefinementTrainer(self.cfg.ebt_refinement)
            trainer.run(examples)
            self.logger.log("EBTModelRetrained", {
                "total_examples": len(examples),
                "dimensions": list(set(e["dimension"] for e in examples))
            })