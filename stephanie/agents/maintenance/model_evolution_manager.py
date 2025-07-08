# stephanie/agents/model_evolution_manager.py

from stephanie.agents.base_agent import BaseAgent
import json

class ModelEvolutionManager(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_dir = cfg.get("model_dir", "models")
        self.min_improvement = cfg.get("min_improvement", 0.05)  # 5% improvement threshold

    def get_best_model(self, model_type: str, target_type: str, dimension: str):
        """Returns the current best model version for a dimension"""
        query = """
        SELECT version, performance 
        FROM model_versions 
        WHERE model_type = :model_type
          AND target_type = :target_type
          AND dimension = :dimension
          AND active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
        """
        result = self.memory.db.execute(query, {
            "model_type": model_type,
            "target_type": target_type,
            "dimension": dimension
        }).fetchone()
        
        if result:
            return {
                "version": result.version,
                "performance": json.loads(result.performance) if result.performance else {}
            }
        return None

    def log_model_version(self, model_type: str, target_type: str, dimension: str, version: str, performance: dict):
        """Record a new model version in the registry"""
        query = """
        INSERT INTO model_versions (
            model_type, target_type, dimension, version, performance, active
        ) VALUES (
            :model_type, :target_type, :dimension, :version, :performance, FALSE
        ) RETURNING id
        """
        result = self.memory.db.execute(query, {
            "model_type": model_type,
            "target_type": target_type,
            "dimension": dimension,
            "version": version,
            "performance": json.dumps(performance)
        }).fetchone()
        
        self.logger.log("ModelVersionLogged", {
            "model_type": model_type,
            "dimension": dimension,
            "version": version,
            "performance": performance
        })
        return result.id

    def promote_model_version(self, model_id: int):
        """Mark a model as active and deprecate previous active models"""
        query = """
        UPDATE model_versions 
        SET active = FALSE 
        WHERE id != :id 
          AND model_type = (SELECT model_type FROM model_versions WHERE id = :id)
          AND target_type = (SELECT target_type FROM model_versions WHERE id = :id)
          AND dimension = (SELECT dimension FROM model_versions WHERE id = :id)
        """
        self.memory.db.execute(query, {"id": model_id})
        
        query = """
        UPDATE model_versions 
        SET active = TRUE 
        WHERE id = :id
        """
        self.memory.db.execute(query, {"id": model_id})
        
        self.logger.log("ModelVersionPromoted", {"model_id": model_id})

    def check_model_performance(self, new_perf: dict, old_perf: dict) -> bool:
        """Compare two model versions to see if new one is better"""
        if not old_perf:
            return True  # no baseline, accept new model
        
        # Compare based on metrics (e.g., lower loss = better)
        new_loss = new_perf.get("validation_loss", float('inf'))
        old_loss = old_perf.get("validation_loss", float('inf'))
        
        # Accept if improvement exceeds threshold
        return (old_loss - new_loss) / old_loss > self.min_improvement