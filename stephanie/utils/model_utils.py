
import os
import re

def get_model_path(model_type: str, target_type: str, dimension: str, version: str = "v1"):
    return f"models/{model_type}/{target_type}/{dimension}_{version}"

def discover_saved_dimensions(model_type: str, target_type: str, model_dir: str = "models") -> list:
    """
    Discover saved dimensions for a given model and target type.
    Filters out scalers and metadata artifacts.
    """
    path = os.path.join(model_dir, model_type, target_type)
    if not os.path.exists(path):
        print(f"[discover_saved_dimensions] Path {path} does not exist.")
        return []

    dimension_names = set()

    for filename in os.listdir(path):
        # Ignore scalers, tuners, and meta
        if any(ex in filename for ex in ["_scaler", ".tuner", ".meta", ".json"]):
            continue

        # Match patterns for EBT, MRQ, SVM
        if filename.endswith(".pt") or filename.endswith(".joblib"):
            # Extract base name (e.g., alignment_v1.pt -> alignment)
            base = filename.split("_v")[0]  # remove _v1 suffix
            base = base.replace(".joblib", "").replace(".pt", "")
            dimension_names.add(base)

    return sorted(dimension_names)

def get_svm_file_paths(model_type, target_type, dim):
    base = get_model_path(model_type, target_type, dim)
    return {
        "model": base + ".joblib",
        "scaler": base + "_scaler.joblib",
        "tuner": base + ".tuner.json",
        "meta": base + ".meta.json"
    }