import numpy as np
from sentence_transformers import SentenceTransformer

from log import log

_model: SentenceTransformer | None = None


def load():
    global _model
    log("memory", "loading embedding model...")
    _model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    log("memory", "embedding model ready")


def encode_query(text: str) -> np.ndarray:
    assert _model is not None, "call load() first"
    return np.array(_model.encode(text, prompt_name="query", normalize_embeddings=True))


def encode_doc(text: str) -> np.ndarray:
    assert _model is not None, "call load() first"
    return np.array(_model.encode(text, prompt_name="document", normalize_embeddings=True))
