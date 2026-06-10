"""
HybridSearch: semantic + keyword + metadata scoring.

Returns List[Tuple[Assessment, float]] — callers use the score for
relevance gating.

Key fix: stopword filtering on keyword overlap so short queries like
"senior software engineer" aren't drowned out by generic tokens.
Query expansion adds role synonyms to boost recall for common roles.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Set, Tuple

from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

W_SEM = float(os.getenv("WEIGHT_SEMANTIC", "0.50"))  # raised: semantic is most reliable
W_KEY = float(os.getenv("WEIGHT_KEYWORD",  "0.20"))  # lowered: stopwords were inflating this
W_JL  = float(os.getenv("WEIGHT_JOB_LEVEL","0.15"))
W_TYP = float(os.getenv("WEIGHT_TYPE",     "0.10"))
W_DUR = float(os.getenv("WEIGHT_DURATION", "0.05"))

_SENIORITY_MAP = {
    "intern":    ["entry-level", "graduate"],
    "entry":     ["entry-level", "graduate"],
    "junior":    ["entry-level", "graduate"],
    "graduate":  ["entry-level", "graduate"],
    "mid":       ["mid-professional"],
    "senior":    ["manager", "mid-professional", "professional individual contributor"],
    "lead":      ["manager", "director"],
    "manager":   ["manager", "front line manager"],
    "director":  ["director"],
    "executive": ["executive", "director"],
    "vp":        ["executive", "director"],
    "c-level":   ["executive"],
}

# Common words that carry no discriminating signal for role matching
_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "of", "in", "to", "is", "are",
    "it", "this", "that", "with", "on", "at", "by", "as", "be", "was",
    "has", "have", "had", "not", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might", "shall",
    # hiring-context words that appear in EVERY query and add no discrimination
    "hiring", "hire", "assessment", "assess", "role", "position", "job",
    "candidate", "need", "want", "looking", "find", "test", "evaluate",
}

# Role synonym expansion — maps common role terms to related domain words
# so "software engineer" also matches tests about "programming", "coding", etc.
_ROLE_EXPANSIONS: Dict[str, List[str]] = {
    "software engineer":   ["programming", "coding", "developer", "software development"],
    "software developer":  ["programming", "coding", "engineer", "software development"],
    "data engineer":       ["sql", "database", "data pipeline", "etl", "data"],
    "data scientist":      ["machine learning", "statistics", "python", "data analysis"],
    "frontend":            ["javascript", "react", "angular", "web", "html", "css"],
    "backend":             ["api", "server", "database", "microservice"],
    "devops":              ["cloud", "linux", "deployment", "infrastructure", "aws", "azure"],
    "java":                ["java", "spring", "backend", "object oriented"],
    "python":              ["python", "scripting", "data", "automation"],
    "manager":             ["leadership", "team management", "people management"],
    "sales":               ["customer", "negotiation", "crm", "revenue"],
    "hr":                  ["recruitment", "human resources", "talent", "people"],
    "finance":             ["accounting", "financial", "excel", "reporting"],
    "customer service":    ["customer support", "communication", "service"],
    "analyst":             ["analysis", "reporting", "excel", "sql", "data"],
}


def _tokenize(text: str) -> Set[str]:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    return tokens - _STOPWORDS


def _expand_query(query: str) -> str:
    """Add synonym terms to query so keyword overlap catches more relevant tests."""
    q_lower = query.lower()
    extras = []
    for role_key, synonyms in _ROLE_EXPANSIONS.items():
        if role_key in q_lower:
            extras.extend(synonyms)
    if extras:
        return query + " " + " ".join(extras)
    return query


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
        pre_filter_k: int = 50,
    ) -> List[Tuple[Assessment, float]]:
        """
        Returns up to top_k (Assessment, hybrid_score) tuples, sorted descending.
        """
        # Expand query with role synonyms before embedding
        expanded_query = _expand_query(query)

        # 1. Semantic retrieval on expanded query
        sem_results: List[Tuple[Assessment, float]] = self.vector_store.search(
            expanded_query, top_k=pre_filter_k
        )
        if not sem_results:
            sem_results = [(a, 0.0) for a in self.catalog.all()]

        sem_map = {a.entity_id: score for a, score in sem_results}
        candidates = [a for a, _ in sem_results]

        # 2. Hard filters
        if remote_only:
            candidates = [a for a in candidates if a.remote_testing]
        if max_duration is not None:
            candidates = [
                a for a in candidates
                if a.duration_minutes is None or a.duration_minutes <= max_duration
            ]
        if not candidates:
            candidates = [a for a, _ in sem_results]  # relax if filters emptied list

        # 3. Score
        query_tokens = _tokenize(expanded_query)
        job_levels = _seniority_to_job_levels(seniority)
        desired_types_set = {t.upper() for t in (desired_types or [])}

        scored: List[Tuple[Assessment, float]] = []
        for a in candidates:
            s_sem = sem_map.get(a.entity_id, 0.0)

            # Keyword: Jaccard on stopword-filtered tokens
            a_tokens = _tokenize(a.embed_text)
            union = len(query_tokens | a_tokens)
            s_key = len(query_tokens & a_tokens) / max(union, 1)

            # Job level
            if job_levels and a.job_levels:
                jl_lower = {j.lower() for j in a.job_levels}
                matches = sum(1 for j in job_levels if j.lower() in jl_lower)
                s_jl = matches / len(job_levels)
            elif not job_levels:
                s_jl = 0.5
            else:
                s_jl = 0.0

            # Test type preference
            s_typ = (1.0 if a.test_type in desired_types_set else 0.0) \
                    if desired_types_set else 0.5

            # Duration fit
            s_dur = max(0.0, 1.0 - a.duration_minutes / 120.0) \
                    if a.duration_minutes else 0.5

            final = (
                W_SEM * s_sem
                + W_KEY * s_key
                + W_JL  * s_jl
                + W_TYP * s_typ
                + W_DUR * s_dur
            )
            scored.append((a, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]