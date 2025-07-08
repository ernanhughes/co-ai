import torch
import os
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.model.mrq_model import MRQModel
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.evaluator.hypothesis_value_predictor import HypothesisValuePredictor
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.model_utils import get_model_path, discover_saved_dimensions
from stephanie.utils.file_utils import load_json

class DocumentMRQInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "mrq"
        self.target_type = "document"
        self.dimensions = cfg.get("dimensions", [])
        self.models = {}
        self.model_meta = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.dimensions:
            self.dimensions = discover_saved_dimensions(
                model_type=self.model_type, target_type=self.target_type
            )

        self.logger.log(
            "DocumentMRQInferenceAgentInitialized",
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "dimensions": self.dimensions,
                "device": str(self.device),
            },
        )

        for dim in self.dimensions:
            model_path = f"{get_model_path(self.model_type, self.target_type, dim)}"
            encoder_path = f"{model_path}_encoder.pt"
            predictor_path = f"{model_path}.pt"
            meta_path = f"{model_path}.meta.json"

            self.logger.log("LoadingModelPaths", {
                "dimension": dim,
                "encoder_path": encoder_path,
                "predictor_path": predictor_path
            })
            encoder = TextEncoder()
            predictor = HypothesisValuePredictor(512, 1024)
            model = MRQModel(encoder, predictor, device=self.device)
            model.load_weights(encoder_path, predictor_path)
            self.models[dim] = model

            if os.path.exists(meta_path):
                self.model_meta[dim] = load_json(meta_path)
            else:
                self.model_meta[dim] = {"min": 0, "max": 100}

        self.logger.log("AllMRQModelsLoaded", {"dimensions": self.dimensions})

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        for doc in context.get(self.input_key, []):
            doc_id = doc.get("id")
            self.logger.log("MRQScoringStarted", {"document_id": doc_id})

            scorable = Scorable(
                id=doc_id, text=doc.get("text", ""), target_type=TargetType.DOCUMENT
            )

            dimension_scores = {}
            for dim, model in self.models.items():
                q_value = model.predict(goal_text, scorable.text, self.memory.embedding)
                meta = self.model_meta.get(dim, {})
                min_val = meta.get("min", 0)
                max_val = meta.get("max", 100)

                normalized = torch.sigmoid(torch.tensor(q_value)).item()
                scaled_score = normalized * (max_val - min_val) + min_val

                dimension_scores[dim] = round(scaled_score, 4)

                self.logger.log("MRQScoreComputed", {
                    "document_id": doc_id,
                    "dimension": dim,
                    "q_value": round(q_value, 4),
                    "normalized_score": normalized,
                    "final_score": scaled_score,
                })

            results.append({"scorable": scorable.to_dict(), "scores": dimension_scores})

            self.logger.log("MRQScoringFinished", {
                "document_id": doc_id,
                "scores": dimension_scores,
                "dimensions_scored": list(dimension_scores.keys()),
            })

        context[self.output_key] = results
        self.logger.log("MRQInferenceCompleted", {"total_documents_scored": len(results)})
        return context
