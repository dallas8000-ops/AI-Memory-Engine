"""Deterministic stub embedder for burn testing (no model download needed)."""
import hashlib
import numpy as np
from memory_engine.embedder import Embedder

def _stub_embed(self, text):
    vec = np.zeros(384, dtype=np.float32)
    for tok in str(text).lower().split():
        seed = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        vec += np.random.default_rng(seed).standard_normal(384).astype(np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n else vec

Embedder.embed = _stub_embed
