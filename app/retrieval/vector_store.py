"""
VectorStore: builds and queries a FAISS index.

Embeddings use the HuggingFace Inference API instead of a local
SentenceTransformer model. This keeps memory under 150MB on Render's
free tier (local torch+sentence-transformers needs ~1.5GB and causes OOM).

The FAISS index is built once and persisted to disk. On subsequent
startups it loads from disk in <1s with zero API calls.
"""

import logging
import os
import pickle
import time
from typing import List, Tuple

import httpx
import numpy as np

from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore

logger = logging.getLogger(__name__)

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "../../data/embeddings.faiss")
_META_PATH  = os.path.join(os.path.dirname(__file__), "../../data/metadata.pkl")

_HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)
_HF_TOKEN   = os.getenv("HF_TOKEN", "")   # optional but increases rate limits
_BATCH_SIZE = 32


class VectorStore:
    def __init__(self, catalog: CatalogStore):
        self.catalog = catalog
        self._index  = None
        self._ids: List[str] = []
        # _model kept for API compatibility with any code that checks it
        self._model  = "hf-api"

    # ── Public API ────────────────────────────────────────────────────────────

    def load_or_build(self) -> None:
        """Called once at startup. Loads index from disk or builds via HF API."""
        if os.path.exists(_INDEX_PATH) and os.path.exists(_META_PATH):
            self._load()
        else:
            self._build()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Assessment, float]]:
        """Return (assessment, cosine_similarity) sorted descending."""
        if self._index is None or not self._ids:
            return []

        vec = self._embed_batch([query])[0]
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

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed texts via HF Inference API in batches. Returns L2-normalised vectors."""
        headers = {"Content-Type": "application/json"}
        if _HF_TOKEN:
            headers["Authorization"] = f"Bearer {_HF_TOKEN}"

        all_vecs = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            vecs  = self._call_hf_api(batch, headers)
            all_vecs.append(vecs)

        combined = np.vstack(all_vecs).astype("float32")
        norms    = np.linalg.norm(combined, axis=1, keepdims=True) + 1e-10
        return combined / norms

    def _call_hf_api(self, texts: List[str], headers: dict) -> np.ndarray:
        """Single HF API call with one retry on 503 (model still loading)."""
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        for attempt in range(2):
            try:
                resp = httpx.post(
                    _HF_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                if resp.status_code == 503 and attempt == 0:
                    logger.warning("HF model loading (503) — retrying in 20s …")
                    time.sleep(20)
                    continue
                resp.raise_for_status()
                return np.array(resp.json(), dtype="float32")
            except httpx.HTTPStatusError as e:
                logger.error("HF API HTTP error: %s", e)
                raise
            except Exception as e:
                logger.error("HF API call failed: %s", e)
                raise

        raise RuntimeError("HF Inference API failed after retry.")

    # ── Index build / load ────────────────────────────────────────────────────

    def _build(self) -> None:
        import faiss

        assessments = self.catalog.all()
        if not assessments:
            logger.warning("Catalog empty — skipping index build.")
            return

        logger.info("Embedding %d assessments via HF API …", len(assessments))
        texts     = [a.embed_text for a in assessments]
        self._ids = [a.entity_id  for a in assessments]

        vecs = self._embed_batch(texts)
        dim  = vecs.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)

        os.makedirs(os.path.dirname(_INDEX_PATH), exist_ok=True)
        faiss.write_index(self._index, _INDEX_PATH)
        with open(_META_PATH, "wb") as f:
            pickle.dump(self._ids, f)

        logger.info("Index built and saved (%d vectors, dim=%d).", len(self._ids), dim)

    def _load(self) -> None:
        import faiss

        logger.info("Loading existing FAISS index …")
        self._index = faiss.read_index(_INDEX_PATH)
        with open(_META_PATH, "rb") as f:
            self._ids = pickle.load(f)
        logger.info("Index loaded (%d vectors).", len(self._ids))