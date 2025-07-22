# In your analysis agent
from stephanie.agents.base_agent import BaseAgent
from stephanie.analysis.policy_analyzer import PolicyAnalyzer

class PolicyAnalysisAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.dimensions = cfg.get("dimensions", ["alignment", "clarity", "novelty"])
        self.analyzer = PolicyAnalyzer(memory.session, logger)

    async def run(self, context: dict) -> dict:
        reports = {}
        
        pipeline_run_id = context.get("pipeline_run_id", None)
        for dim in self.dimensions:

            report = self.analyzer.generate_policy_report(dim, pipeline_run_id=pipeline_run_id)
            reports[dim] = report
            
            # Log insights
            for insight in report.get("insights", []):
                self.logger.log("PolicyInsight", {
                    "dimension": dim,
                    "insight": insight
                })
                
            # Generate visualization
            if self.cfg.get("generate_visualization", True):
                viz_paths = self.analyzer.visualize_policy(dim)
                if viz_paths:
                    report["visualizations"] = viz_paths
        
                    
        # context["policy_analysis"] = reports
        return context