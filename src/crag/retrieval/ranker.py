"""Semantic reranking of on-demand chunks with a disk-backed embedding cache.

The hybrid retrieval strategy embeds only the chunks produced for the current
query (never the whole corpus). Chunk embeddings are content-addressed on disk
(``<chunk_id>.npy``) so recurring topics get progressively faster.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.crag.models import Chunk

logger = logging.getLogger(__name__)


class SemanticRanker:
    def __init__(self, model_name: str, cache_dir: Path) -> None:
        self._model_name = model_name
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # lazy: loading sentence-transformers is expensive

    def _get_model(self):
        if self._model is None:
            # Imported lazily so unit tests that never rank don't pay the cost.
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model %s (HF_HOME cache)", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # ------------------------------------------------------------- embedding
    def _cache_path(self, chunk: Chunk) -> Path:
        return self._cache_dir / f"{chunk.chunk_id}.npy"

    def _embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        vectors: list[np.ndarray | None] = [None] * len(chunks)
        missing_idx: list[int] = []

        for i, chunk in enumerate(chunks):
            path = self._cache_path(chunk)
            if path.exists():
                try:
                    vectors[i] = np.load(path)
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
            missing_idx.append(i)

        if missing_idx:
            texts = [chunks[i].text for i in missing_idx]
            fresh = self._get_model().encode(texts, normalize_embeddings=True)
            for i, vec in zip(missing_idx, fresh):
                vectors[i] = np.asarray(vec, dtype=np.float32)
                try:
                    np.save(self._cache_path(chunks[i]), vectors[i])
                except OSError as exc:
                    logger.warning("Could not cache embedding: %s", exc)

        return np.vstack(vectors)

    # --------------------------------------------------------------- ranking
    def rank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return the ``top_n`` chunks by cosine similarity to the query."""
        if not chunks:
            return []
        query_vec = self._get_model().encode([query], normalize_embeddings=True)[0]
        chunk_matrix = self._embed_chunks(chunks)
        scores = chunk_matrix @ query_vec  # cosine (embeddings are normalized)

        for chunk, score in zip(chunks, scores):
            chunk.semantic_score = float(score)
        ranked = sorted(chunks, key=lambda c: c.semantic_score, reverse=True)
        return ranked[:top_n]
