import os
import numpy as np
from joblib import load

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.model_utils import discover_saved_dimensions, get_svm_file_paths
from stephanie.utils.file_utils import load_json
from stephanie.scoring.transforms.regression_tuner import RegressionTuner


class DocumentSVMInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "svm"
        self.target_type = "document"
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}  # dim -> (scaler, model)
        self.model_meta = {}  # dim -> {min, max}
        self.tuners = {}  # dim -> RegressionTuner

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type,
                target_type=self.target_type
            )

        self.logger.log("DocumentSVMInferenceInitialized", {
            "dimensions": self.dimensions
        })

        for dim in self.dimensions:
            paths = get_svm_file_paths(self.model_type, self.target_type, dim)
            scaler_path = paths["scaler"]
            model_file = paths["model"]
            meta_path = paths["meta"]

            self.logger.log("LoadingSVMModel", {"dimension": dim, "model": model_file})

            self.models[dim] = (load(scaler_path), load(model_file))
            self.model_meta[dim] = load_json(meta_path) if os.path.exists(meta_path) else {"min_score": 0, "max_score": 100}
            self.tuners[dim] = RegressionTuner(dimension=dim, logger=logger)
            self.tuners[dim].load(paths["tuner"])

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        for doc in context.get(self.input_key, []):
            doc_id = doc.get("id")
            self.logger.log("SVMScoringStarted", {"document_id": doc_id})

            scorable = Scorable(
                id=doc_id,
                text=doc.get("text", ""),
                target_type=TargetType.DOCUMENT
            )

            ctx_emb = self.memory.embedding.get_or_create(goal_text)
            doc_emb = self.memory.embedding.get_or_create(scorable.text)
            feature = np.array(ctx_emb + doc_emb).reshape(1, -1)

            dimension_scores = {}
            for dim, (scaler, model) in self.models.items():
                X_scaled = scaler.transform(feature)
                raw_score = model.predict(X_scaled)[0]

                tuned_score = self.tuners[dim].transform(raw_score)

                meta = self.model_meta.get(dim, {"min_score": 0, "max_score": 100})
                min_s, max_s = meta["min_score"], meta["max_score"]
                final_score = max(min(tuned_score, max_s), min_s)

                dimension_scores[dim] = round(final_score, 4)
                self.logger.log("SVMScoreComputed", {
                    "document_id": doc_id,
                    "dimension": dim,
                    "raw_score": round(raw_score, 4),
                    "tuned_score": round(tuned_score, 4),
                    "final_score": round(final_score, 4)
                })

            results.append({
                "scorable": scorable.to_dict(),
                "scores": dimension_scores
            })

            self.logger.log("SVMScoringFinished", {
                "document_id": doc_id,
                "scores": dimension_scores,
                "dimensions_scored": list(dimension_scores.keys())
            })

        context[self.output_key] = results
        self.logger.log("SVMInferenceCompleted", {"total_documents_scored": len(results)})
        return context
