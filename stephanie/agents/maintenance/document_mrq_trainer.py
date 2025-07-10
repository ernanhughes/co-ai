# stephanie/agents/maintenance/document_mrq_trainer.py

import os
import torch
from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.mrq.encoder import TextEncoder
from stephanie.scoring.mrq.trainer_engine import MRQTrainerEngine
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.document_value_predictor import ValuePredictor
from stephanie.utils.model_utils import get_model_path
from stephanie.utils.file_utils import save_json


class MRQTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "mrq")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        builder = PreferencePairBuilder(db=self.memory.session, logger=self.logger)
        training_pairs_by_dim = builder.get_training_pairs_by_dimension(goal=goal_text)

        contrast_pairs = [
            {
                "title": item["title"],
                "output_a": item["output_a"],
                "output_b": item["output_b"],
                "value_a": item["value_a"],
                "value_b": item["value_b"],
                "dimension": dim
            }
            for dim, pairs in training_pairs_by_dim.items()
            for item in pairs
        ]

        self.logger.log("DocumentPairBuilderComplete", {
            "dimensions": list(training_pairs_by_dim.keys()),
            "total_pairs": len(contrast_pairs)
        })

        trainer = MRQTrainerEngine(
            memory=self.memory,
            logger=self.logger,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        config = {
            "epochs": 10,
            "lr": 1e-4,
            "patience": 2,
            "min_delta": 0.001
        }

        assert contrast_pairs, "No contrast pairs found"

        trained_encoders, trained_models, regression_tuners = trainer.train_all(
            contrast_pairs, cfg=config
        )

        for dim in trained_models:
            model_path = get_model_path(
                self.model_path,
                self.model_type,
                self.target_type,
                dim,
                self.model_version,
            )
            os.makedirs(model_path, exist_ok=True)

            predictor_path = os.path.join(model_path, f"{dim}.pt")
            encoder_path = os.path.join(model_path, f"{dim}_encoder.pt")
            tuner_path = os.path.join(model_path, f"{dim}_model.tuner.json")
            meta_path = os.path.join(model_path, f"{dim}.meta.json")

            # Save model weights
            torch.save(trained_models[dim], predictor_path)
            self.logger.log("ModelSaved", {"dimension": dim, "path": predictor_path})

            # Save encoder weights
            encoder_state = trained_encoders.get(dim)
            if encoder_state:
                torch.save(encoder_state, encoder_path)
                self.logger.log("EncoderSaved", {"dimension": dim, "path": encoder_path})
            else:
                self.logger.log("EncoderMissing", {"dimension": dim})

            # Save regression tuner
            tuner = regression_tuners.get(dim)
            if tuner:
                tuner.save(tuner_path)

            # Save normalization metadata
            values = [
                (p["value_a"], p["value_b"])
                for p in contrast_pairs if p["dimension"] == dim
            ]
            flat_values = [v for pair in values for v in pair]
            save_json({
                "min_score": float(min(flat_values)),
                "max_score": float(max(flat_values))
            }, meta_path)

            self.logger.log("DocumentModelSaved", {
                "dimension": dim,
                "model": predictor_path,
                "encoder": encoder_path,
                "tuner": tuner_path,
                "meta": meta_path
            })

        context[self.output_key] = training_pairs_by_dim
        return context