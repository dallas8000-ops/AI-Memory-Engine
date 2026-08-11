"""Text -> vector embedding using sentence-transformers (lazy-loaded)."""
from __future__ import annotations

import numpy as np

from . import config


class Embedder:
    """Wraps a sentence-transformers model. The model is loaded on first use
    so importing this module stays fast."""

    def __init__(self, model_name: str = config.EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return config.EMBED_DIM

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string into a float32 vector."""
        model = self._load()
        vec = model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
