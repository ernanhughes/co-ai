
import os
import re

def get_model_path(model_type: str, target_type: str, dimension: str, version: str = "v1"):
    return f"models/{model_type}/{target_type}/{dimension}_{version}"

def discover_saved_dimensions(model_type: str, target_type: str, model_dir: str = "models") -> list:
    """
    Scan the model directory and extract dimension names from filenames like 'dimension_v1.pt'
    """
    path = os.path.join(model_dir, model_type, target_type)
    if not os.path.exists(path):
        return []

    dimension_names = []
    for filename in os.listdir(path):
        match = re.match(r"(.+)_v\d+\.pt$", filename)
        if match:
            dimension_names.append(match.group(1))

    return sorted(dimension_names)
