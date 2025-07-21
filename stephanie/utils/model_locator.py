import os
from typing import Dict, List


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

    @staticmethod
    def list_available_models(root_dir="models") -> List[str]:
        available = []
        for embedding in os.listdir(root_dir):
            embedding_path = os.path.join(root_dir, embedding)
            if not os.path.isdir(embedding_path):
                continue
            for model_type in os.listdir(embedding_path):
                type_path = os.path.join(embedding_path, model_type)
                for target_type in os.listdir(type_path):
                    target_path = os.path.join(type_path, target_type)
                    for dimension in os.listdir(target_path):
                        dim_path = os.path.join(target_path, dimension)
                        for version in os.listdir(dim_path):
                            version_path = os.path.join(dim_path, version)
                            available.append(
                                f"{embedding}/{model_type}/{target_type}/{dimension}/{version}"
                            )
        return sorted(available)

    @staticmethod
    def find_best_model_per_dimension(root_dir="models") -> Dict[str, str]:
        best = {}
        for model_path in ModelLocator.list_available_models(root_dir):
            parts = model_path.split("/")
            if len(parts) < 5:
                continue
            _, model_type, target_type, dimension, version = parts[-5:]
            if dimension not in best or version > best[dimension].split("/")[-1]:
                best[dimension] = model_path
        return best
