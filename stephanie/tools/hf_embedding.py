# stephanie/embeddings/huggingface_embedder.py

from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    _model_instance = None  # class-level singleton

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_name = cfg.get("hf_model_name", "Qwen/Qwen3-Embedding-8B")

        if HuggingFaceEmbedder._model_instance is None:
            HuggingFaceEmbedder._model_instance = SentenceTransformer(self.model_name)

        self.model = HuggingFaceEmbedder._model_instance

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            return []

        text = f"passage: {text.strip()}" if "e5" in self.model_name.lower() else text
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        processed = [f"passage: {t.strip()}" if "e5" in self.model_name.lower() else t for t in texts]
        embeddings = self.model.encode(processed, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()



_model_instance = None


def load_model(model_name="Qwen/Qwen3-Embedding-8B"):
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(model_name)
    return _model_instance


def get_embedding(text: str, cfg: dict) -> list[float]:
    """
    Embed a single piece of text using HuggingFace model.
    """
    model_name = cfg.get("hf_model_name", "Qwen/Qwen3-Embedding-8B")
    model = load_model(model_name)

    if not text.strip():
        return []

    # Some E5 models expect prefixes
    if "e5" in model_name.lower():
        text = f"passage: {text.strip()}"

    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def batch_embed(texts: list[str], cfg: dict) -> list[list[float]]:
    model_name = cfg.get("hf_model_name", "intfloat/e5-large-v2")
    model = load_model(model_name)

    texts = [f"passage: {t.strip()}" if "e5" in model_name.lower() else t for t in texts]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()
