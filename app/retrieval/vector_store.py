"""
VectorStore: embeds assessments and builds a FAISS index.

Key fix: SentenceTransformer is imported and loaded ONCE during
load_or_build() at startup — never lazily inside a request handler.
This prevents MemoryError from repeated heavy imports under load.
"""

import logging
import os
import pickle
from typing import List, Tuple

import numpy as np

from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore

logger = logging.getLogger(__name__)

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "../../data/embeddings.faiss")
_META_PATH  = os.path.join(os.path.dirname(__file__), "../../data/metadata.pkl")
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, catalog: CatalogStore):
        self.catalog = catalog
        self._index = None
        self._ids: List[str] = []
        self._model = None          # set once in load_or_build()

    # ── Public API ──────────────────────────────────────────────────────────

    def load_or_build(self) -> None:
        """
        Called once at application startup.
        Loads the embedding model first, then loads or builds the FAISS index.
        Doing this here (not inside search()) keeps the request path memory-safe.
        """
        self._load_model()          # import + load weights exactly once
        if os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH):
            self._load()
        else:
            self._build()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Assessment, float]]:
        """Return (assessment, cosine_similarity) sorted descending."""
        if self._index is None or not self._ids:
            return []

        if self._model is None:
            # Defensive: should not happen after load_or_build(), but handle gracefully
            logger.error("Embedding model not loaded — returning empty results")
            return []

        vec = self._embed([query])[0]
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        vec = np.array([vec], dtype="float32")

        k = min(top_k, len(self._ids))
        distances, indices = self._index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            assessment = self.catalog.get_by_id(self._ids[idx])
            if assessment:
                results.append((assessment, float(dist)))
        return results

    # ── Internal ────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Import sentence_transformers and load weights. Called once at startup."""
        if self._model is not None:
            return
        try:
            # Import here so the MemoryError (if any) surfaces at startup with
            # a clear message, not silently inside a request handler.
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model %s …", _MODEL_NAME)
            self._model = SentenceTransformer(_MODEL_NAME)
            logger.info("Embedding model loaded.")
        except MemoryError:
            logger.critical(
                "MemoryError while loading SentenceTransformer. "
                "Free RAM and restart, or switch to a lighter model."
            )
            raise
        except Exception as e:
            logger.critical("Failed to load embedding model: %s", e)
            raise

    def _embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,          # explicit batch size avoids OOM on large inputs
        )
        return np.array(vecs, dtype="float32")

    def _build(self) -> None:
        import faiss

        assessments = self.catalog.all()
        if not assessments:
            logger.warning("Catalog is empty — skipping index build")
            return

        logger.info("Embedding %d assessments …", len(assessments))
        texts  = [a.embed_text  for a in assessments]
        self._ids = [a.entity_id for a in assessments]

        vecs = self._embed(texts)
        dim  = vecs.shape[1]

        self._index = faiss.IndexFlatIP(dim)  # Inner Product on L2-norm = cosine
        self._index.add(vecs)

        os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
        faiss.write_index(self._index, _INDEX_PATH)
        with open(_META_PATH, "wb") as f:
            pickle.dump(self._ids, f)

        logger.info("Index built and saved (%d vectors, dim=%d)", len(self._ids), dim)

    def _load(self) -> None:
        import faiss

        logger.info("Loading existing FAISS index …")
        self._index = faiss.read_index(_INDEX_PATH)
        with open(_META_PATH, "rb") as f:
            self._ids = pickle.load(f)
        logger.info("Index loaded (%d vectors)", len(self._ids))