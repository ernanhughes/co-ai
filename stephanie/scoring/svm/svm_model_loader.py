import os
from typing import Dict, Tuple

import numpy as np
from joblib import load
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from stephanie.utils.file_utils import load_json
from stephanie.scoring.transforms.regression_tuner import RegressionTuner
from stephanie.utils.model_utils import get_svm_file_paths, discover_saved_dimensions


class SVMModelLoader:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.model_path = cfg.get("model_path", "models")
        self.model_type = cfg.get("model_type", "svm")
        self.target_type = cfg.get("target_type", "document")
        self.model_version = cfg.get("model_version", "v1")

    def load_all(self, dimensions: list[str] = None) -> Dict[str, Tuple[StandardScaler, SVR]]:
        """
        Load all SVM models and scalers for the given dimensions.
        """
        if not dimensions:
            dimensions = discover_saved_dimensions(
                model_type=self.model_type,
                target_type=self.target_type,
            )

        models = {}
        for dim in dimensions:
            scaler, model = self.load_dimension(dim)
            models[dim] = (scaler, model)

        return models

    def load_dimension(self, dim: str) -> Tuple[StandardScaler, SVR]:
        paths = get_svm_file_paths(
            self.model_path,
            self.model_type,
            self.target_type,
            dim,
            self.model_version,
        )

        if self.logger:
            self.logger.log("LoadingSVMModel", {"dimension": dim, "model": paths["model"]})

        scaler = load(paths["scaler"])
        model = load(paths["model"])
        return scaler, model

    def load_tuner(self, dim: str) -> RegressionTuner:
        paths = get_svm_file_paths(
            self.model_path,
            self.model_type,
            self.target_type,
            dim,
            self.model_version,
        )
        tuner = RegressionTuner(dimension=dim, logger=self.logger)
        tuner.load(paths["tuner"])
        return tuner

    def load_meta(self, dim: str) -> dict:
        paths = get_svm_file_paths(
            self.model_path,
            self.model_type,
            self.target_type,
            dim,
            self.model_version,
        )
        return (
            load_json(paths["meta"])
            if os.path.exists(paths["meta"])
            else {"min_score": 0, "max_score": 100}
        )
