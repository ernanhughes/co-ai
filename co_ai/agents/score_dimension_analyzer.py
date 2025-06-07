from co_ai.agents.base import BaseAgent
from collections import defaultdict
from co_ai.constants import PIPELINE_RUN_ID

class ScoreDimensionAnalyzerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.min_count = cfg.get("min_count", 3)

    async def run(self, context: dict):
        pipeline_run_id = context.get(PIPELINE_RUN_ID)
        scores = self.memory.scores.get_by_pipeline_run(pipeline_run_id)

        # Collect dimension scores
        dimension_totals = defaultdict(float)
        dimension_counts = defaultdict(int)

        for score_obj in scores:
            extra_data = score_obj.extra_data or {}
            for k, v in extra_data.items():
                try:
                    v = float(v)
                    dimension_totals[k] += v
                    dimension_counts[k] += 1
                except (ValueError, TypeError):
                    continue

        # Compute averages
        dimension_averages = {
            k: dimension_totals[k] / dimension_counts[k]
            for k in dimension_totals
            if dimension_counts[k] >= self.min_count
        }

        # Log results
        self.logger.log("ScoreDimensionSummary", {
            "pipeline_run_id": pipeline_run_id,
            "dimension_averages": dimension_averages,
            "dimension_counts": dict(dimension_counts),
        })

        # Optionally attach to context for downstream agents
        context["dimension_scores"] = dimension_averages
        return context
