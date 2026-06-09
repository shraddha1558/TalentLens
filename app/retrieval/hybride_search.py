"""
HybridSearch: combines semantic similarity, keyword overlap, job-level match,
test-type match, and duration fit into a single ranked list.

Weights (tunable via env vars):
  WEIGHT_SEMANTIC   = 0.45
  WEIGHT_KEYWORD    = 0.25
  WEIGHT_JOB_LEVEL  = 0.15
  WEIGHT_TYPE       = 0.10
  WEIGHT_DURATION   = 0.05
"""

import os
import re
import logging
from typing import List, Optional, Set

from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

W_SEM = float(os.getenv("WEIGHT_SEMANTIC", "0.45"))
W_KEY = float(os.getenv("WEIGHT_KEYWORD", "0.25"))
W_JL  = float(os.getenv("WEIGHT_JOB_LEVEL", "0.15"))
W_TYP = float(os.getenv("WEIGHT_TYPE", "0.10"))
W_DUR = float(os.getenv("WEIGHT_DURATION", "0.05"))

# Seniority keyword → SHL job_level strings
_SENIORITY_MAP = {
    "intern": ["entry level", "graduate"],
    "entry": ["entry level", "graduate"],
    "junior": ["entry level", "graduate"],
    "graduate": ["entry level", "graduate"],
    "mid": ["mid-professional"],
    "senior": ["manager", "mid-professional"],
    "lead": ["manager", "director"],
    "manager": ["manager"],
    "director": ["director"],
    "executive": ["executive", "director"],
    "vp": ["executive", "director"],
    "c-level": ["executive"],
}


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _seniority_to_job_levels(seniority: Optional[str]) -> List[str]:
    if not seniority:
        return []
    s = seniority.lower()
    for k, v in _SENIORITY_MAP.items():
        if k in s:
            return v
    return []


class HybridSearch:
    def __init__(self, catalog: CatalogStore, vector_store: VectorStore):
        self.catalog = catalog
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        seniority: Optional[str] = None,
        desired_types: Optional[List[str]] = None,
        max_duration: Optional[int] = None,
        remote_only: bool = False,
        top_k: int = 10,
        pre_filter_k: int = 40,
    ) -> List[Assessment]:
        """
        Returns up to top_k Assessment objects ranked by hybrid score.
        """
        # 1. Semantic retrieval (cast a wide net first)
        sem_results = self.vector_store.search(query, top_k=pre_filter_k)
        if not sem_results:
            # Fallback: use all catalog items with zero semantic score
            sem_results = [(a, 0.0) for a in self.catalog.all()]

        sem_map = {a.entity_id: score for a, score in sem_results}
        candidates = [a for a, _ in sem_results]

        # 2. Apply hard metadata filters (remote, duration)
        if remote_only:
            candidates = [a for a in candidates if a.remote_testing]
        if max_duration is not None:
            candidates = [
                a for a in candidates
                if a.duration_minutes is None or a.duration_minutes <= max_duration
            ]

        if not candidates:
            candidates = [a for a, _ in sem_results]  # relax filters if empty

        # 3. Score each candidate
        query_tokens = _tokenize(query)
        job_levels = _seniority_to_job_levels(seniority)
        desired_types_set = {t.upper() for t in (desired_types or [])}

        scored = []
        for a in candidates:
            s_sem = sem_map.get(a.entity_id, 0.0)

            # Keyword overlap (Jaccard-like)
            a_tokens = _tokenize(a.embed_text)
            overlap = len(query_tokens & a_tokens)
            s_key = overlap / max(len(query_tokens | a_tokens), 1)

            # Job level match
            if job_levels and a.job_levels:
                jl_lower = {j.lower() for j in a.job_levels}
                matches = sum(1 for j in job_levels if j.lower() in jl_lower)
                s_jl = matches / len(job_levels)
            elif not job_levels:
                s_jl = 0.5   # neutral when unknown
            else:
                s_jl = 0.0

            # Test type match
            if desired_types_set:
                s_typ = 1.0 if a.test_type in desired_types_set else 0.0
            else:
                s_typ = 0.5  # neutral

            # Duration fit (prefer shorter for screening contexts)
            if a.duration_minutes is not None:
                s_dur = max(0.0, 1.0 - a.duration_minutes / 120.0)
            else:
                s_dur = 0.5

            final = (
                W_SEM * s_sem
                + W_KEY * s_key
                + W_JL  * s_jl
                + W_TYP * s_typ
                + W_DUR * s_dur
            )
            scored.append((a, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, _ in scored[:top_k]]