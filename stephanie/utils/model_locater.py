# stephanie/utils/model_locator.py

import os
from typing import Dict


class ModelLocator:
    def __init__(
        self,
        root_dir: str = "models",
        embedding_type: str = "default",
        model_type: str = "mrq",
        target_type: str = "document",
        dimension: str = "alignment",
        version: str = "v1",
        variant: str = None  # e.g., "sicql"
    ):
        self.root_dir = root_dir
        self.embedding_type = embedding_type
        self.model_type = model_type
        self.target_type = target_type
        self.dimension = dimension
        self.version = version
        self.variant = variant

    @property
    def base_path(self) -> str:
        path = os.path.join(
            self.root_dir,
            self.embedding_type,
            self.model_type,
            self.target_type,
            self.dimension,
            self.version
        )
        return os.path.join(path, self.variant) if self.variant else path

    def model_file(self, suffix: str = ".pt") -> str:
        return os.path.join(self.base_path, f"{self.dimension}{suffix}")

    def encoder_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_encoder.pt")

    def tuner_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}.tuner.json")

    def meta_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}.meta.json")

    def scaler_file(self) -> str:
        return os.path.join(self.base_path, f"{self.dimension}_scaler.joblib")

    def all_files(self) -> Dict[str, str]:
        return {
            "model": self.model_file(".pt" if self.model_type != "svm" else ".joblib"),
            "encoder": self.encoder_file(),
            "meta": self.meta_file(),
            "tuner": self.tuner_file(),
            "scaler": self.scaler_file() if self.model_type == "svm" else None,
        }

    def ensure_dirs(self):
        os.makedirs(self.base_path, exist_ok=True)
