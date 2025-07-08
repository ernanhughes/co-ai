import torch
import os
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.scorable import Scorable
from stephanie.scoring.scorable_factory import TargetType
from stephanie.utils.model_utils import get_model_path, discover_saved_dimensions
from stephanie.scoring.model.ebt_model import EBTModel
from stephanie.utils.file_utils import load_json

class DocumentEBTInferenceAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "ebt"
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
            "DocumentEBTInferenceAgentInitialized",
            {
                "model_type": self.model_type,
                "target_type": self.target_type,
                "dimensions": self.dimensions,
                "device": str(self.device),
            },
        )

        for dim in self.dimensions:
            model_path = f"{get_model_path(self.model_type, self.target_type, dim)}.pt"
            meta_path = model_path.replace(".pt", ".meta.json")

            self.logger.log("LoadingEBTModel", {"dimension": dim, "path": model_path})
            model = self._load_model(model_path)
            self.models[dim] = model

            # Load normalization meta
            if os.path.exists(meta_path):
                self.model_meta[dim] = load_json(meta_path)
            else:
                self.model_meta[dim] = {"min": 40, "max": 100}
        self.logger.log("AllEBTModelsLoaded", {"dimensions": self.dimensions})

    def _load_model(self, path):
        model = EBTModel().to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        results = []

        for doc in context.get(self.input_key, []):
            doc_id = doc.get("id")
            self.logger.log("EBTScoringStarted", {"document_id": doc_id})

            scorable = Scorable(
                id=doc_id, text=doc.get("text", ""), target_type=TargetType.DOCUMENT
            )

            ctx_emb = torch.tensor(self.memory.embedding.get_or_create(goal_text)).to(
                self.device
            )
            doc_emb = torch.tensor(
                self.memory.embedding.get_or_create(scorable.text)
            ).to(self.device)

            dimension_scores = {}
            for dim, model in self.models.items():
                with torch.no_grad():
                    raw_energy = model(ctx_emb, doc_emb).squeeze().cpu().item()
                    meta = self.model_meta.get(dim, {"min": 40, "max": 100})
                    # Normalize energy to [0, 1]
                    normalized_score = torch.sigmoid(torch.tensor(raw_energy)).item()
                    # Then scale to desired range
                    real_score = normalized_score * (meta["max"] - meta["min"]) + meta["min"]
                    dimension_scores[dim] = round(real_score, 4)
                    self.logger.log(
                        "EBTScoreComputed",
                        {
                            "document_id": doc_id,
                            "dimension": dim,
                            "raw_energy": round(raw_energy, 4),
                            "normalized_score": normalized_score,
                            "real_score": real_score,
                        },
                    )

            results.append({"scorable": scorable.to_dict(), "scores": dimension_scores})

            self.logger.log(
                "EBTScoringFinished",
                {
                    "document_id": doc_id,
                    "scores": dimension_scores,
                    "dimensions_scored": list(dimension_scores.keys()),
                },
            )

        context[self.output_key] = results
        self.logger.log(
            "EBTInferenceCompleted", {"total_documents_scored": len(results)}
        )
        return context
