import os
import torch
from stephanie.agents.base_agent import BaseAgent
from stephanie.evaluator.text_encoder import TextEncoder
from stephanie.scoring.document_mrq_trainer import DocumentMRQTrainer
from stephanie.scoring.document_pair_builder import DocumentPreferencePairBuilder
from stephanie.scoring.document_value_predictor import DocumentValuePredictor
from stephanie.utils.model_utils import get_model_path
from stephanie.utils.file_utils import save_json


class DocumentMRQTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_type = "mrq"
        self.target_type = "document"
        self.version = cfg.get("version", "v1")

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")
        builder = DocumentPreferencePairBuilder(db=self.memory.session, logger=self.logger)
        training_pairs = builder.get_training_pairs_by_dimension(goal=goal_text)

        all_contrast_pairs = []

        for dimension, pairs in training_pairs.items():
            for item in pairs:
                all_contrast_pairs.append({
                    "title": item["title"],
                    "output_a": item["output_a"],
                    "output_b": item["output_b"],
                    "value_a": item["value_a"],
                    "value_b": item["value_b"],
                    "dimension": dimension
                })

        self.logger.log("DocumentPairBuilderComplete", {
            "dimensions": list(training_pairs.keys()),
            "total_pairs": sum(len(p) for p in training_pairs.values())
        })

        trainer = DocumentMRQTrainer(
            memory=self.memory,
            logger=self.logger,
            encoder=TextEncoder(),
            value_predictor=DocumentValuePredictor(),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        config = {
            "epochs": 10,
            "lr": 1e-4,
            "patience": 2,
            "min_delta": 0.001
        }

        assert isinstance(all_contrast_pairs, list), "Expected list for contrast pairs"
        assert len(all_contrast_pairs) > 0, "No contrast pairs found"

        trained_encoders, trained_models, regression_tuners = trainer.train_multidimensional_model(
            all_contrast_pairs, cfg=config
        )

        for dim, state_dict in trained_models.items():
            base_path = get_model_path(self.model_type, self.target_type, dim, version=self.version)
            model_dir = os.path.dirname(base_path)
            os.makedirs(model_dir, exist_ok=True)

            predictor_path = f"{base_path}.pt"
            encoder_path = f"{base_path}_encoder.pt"
            tuner_path = f"{base_path}.tuner.json"
            meta_path = f"{base_path}.meta.json"

            # Save predictor            
            torch.save(state_dict, predictor_path)
            self.logger.log("ModelSaved", {"dimension": dim, "path": predictor_path})
            # Save encoder            
            torch.save(trainer.encoder.state_dict(), encoder_path)
            self.logger.log("EncoderSaved", {"dimension": dim, "path": encoder_path})

            # Save tuner
            tuner = regression_tuners.get(dim)
            if tuner:
                tuner.save(tuner_path)

            # Save normalization metadata
            min_score = float(min(p["value_a"] for p in all_contrast_pairs if p["dimension"] == dim))
            max_score = float(max(p["value_b"] for p in all_contrast_pairs if p["dimension"] == dim))
            save_json({"min_score": min_score, "max_score": max_score}, meta_path)

            self.logger.log("DocumentModelSaved", {
                "dimension": dim,
                "model": predictor_path,
                "encoder": encoder_path,
                "tuner": tuner_path,
                "meta": meta_path
            })

        context[self.output_key] = training_pairs
        self.logger.log("DocumentPairBuilderComplete", {
            "dimensions": list(training_pairs.keys()),
            "total_pairs": sum(len(p) for p in training_pairs.values())
        })
        return context
