# stephanie/agents/maintenance/document_svm_trainer.py
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from stephanie.agents.base_agent import BaseAgent
from stephanie.scoring.mrq.preference_pair_builder import PreferencePairBuilder
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.file_utils import save_json
from stephanie.utils.model_utils import get_model_path
from joblib import dump
from stephanie.utils.model_locator import ModelLocator


class SVMTrainerAgent(BaseAgent):
    def __init__(self, cfg, memory=None, logger=None):
        super().__init__(cfg, memory, logger)
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "svm")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")
        self.embedding_type = self.memory.embedding.type


        self.models = {}  # dim -> (scaler, model)
        self.regression_tuners = {}

        self.dimensions = cfg.get("dimensions", [])
        # Initialize tuners and models
        for dim in self.dimensions:
            self._initialize_dimension(dim)
        self.logger.log(
            "SVMTrainerInitialized",
            {
                "dimensions": self.dimensions,
                "model_type": self.model_type,
                "target_type": self.target_type,
                "model_version": self.model_version,
                "model_path": self.model_path,
                "embedding_type": self.embedding_type, 
            },
        )

    def _initialize_dimension(self, dim):
        """Initialize SVM model, scaler, and tuner for each dimension"""
        self.models[dim] = (StandardScaler(), SVR(kernel="linear"))
        self.regression_tuners[dim] = RegressionTuner(
            dimension=dim, logger=self.logger
        )

    async def run(self, context: dict) -> dict:
        goal_text = context.get("goal", {}).get("goal_text")

        builder = PreferencePairBuilder(
            db=self.memory.session, logger=self.logger
        )
        training_pairs = {}
        for dim in self.dimensions:
            pairs = builder.get_training_pairs_by_dimension(goal=goal_text, dim=[dim])
            training_pairs[dim] = pairs.get(dim, [])
       

        for dim, pairs in training_pairs.items():
            self._initialize_dimension(dim)
            if not pairs:
                self.logger.log("SVMNoTrainingPairs", {"dimension": dim})
                continue

            self.logger.log("SVMTrainingStart", {"dimension": dim, "num_pairs": len(pairs)})

            X, y = [], []

            for pair in pairs:
                title = pair["title"]
                for side in ["a", "b"]:
                    output = pair[f"output_{side}"]
                    score = pair[f"value_{side}"]
                    ctx_emb = self.memory.embedding.get_or_create(title)
                    doc_emb = self.memory.embedding.get_or_create(output)
                    feature = np.array(ctx_emb + doc_emb)
                    X.append(feature)
                    y.append(score)

            if len(X) < 2:
                self.logger.log("SVMNotEnoughData", {"dimension": dim})
                continue

            X = np.array(X)
            y = np.array(y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = SVR(kernel="linear")
            model.fit(X_scaled, y)

            # Save using ModelLocator
            locator = ModelLocator(
                root_dir=self.model_path,
                embedding_type=self.embedding_type,
                model_type=self.model_type,
                target_type=self.target_type,
                dimension=dim,
                version=self.model_version,
            )
            save_dir = locator.ensure_dirs()

            dump(scaler, locator.scaler_file())
            dump(model, locator.model_file(suffix=".joblib"))

            meta = {
                "min_score": float(np.min(y)),
                "max_score": float(np.max(y)),
            }
            save_json(locator.meta_file(), meta)

            tuner = self.regression_tuners[dim]
            for i in range(len(X)):
                tuner.train_single(model.predict(X_scaled[i].reshape(1, -1))[0], y[i])
            tuner.save(locator.tuner_file())

            self.logger.log("SVMModelSaved", {"dimension": dim, "path": save_dir})

        context[self.output_key] = training_pairs
        return context
