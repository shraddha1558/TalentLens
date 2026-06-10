"""
CatalogStore: loads shl_catalog.json and exposes fast lookup helpers.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional

from app.models.assessment import Assessment

logger = logging.getLogger(__name__)

_CATALOG_PATH = os.path.join(os.path.dirname(__file__), "../../data/shl_catalog.json")

_KEY_TO_TYPE: Dict[str, str] = {
    "knowledge & skills":            "K",
    "knowledge and skills":          "K",
    "simulations":                   "S",
    "personality & behaviour":       "P",
    "personality and behaviour":     "P",
    "personality & behavior":        "P",
    "personality and behavior":      "P",
    "personality":                   "P",
    "opq":                           "P",
    "ability & aptitude":            "A",
    "ability and aptitude":          "A",
    "ability":                       "A",
    "biodata & situational judgment": "B",
    "biodata & situational judgement":"B",
    "biodata and situational judgement":"B",
    "situational judgment":          "B",
    "situational judgement":         "B",
    "competencies":                  "P",
    "competency":                    "P",
    "development & 360":             "P",
    "assessment exercises":          "S",
}


def _infer_type(keys: List[str], name: str) -> str:
    name_lower = name.lower()
    for k in keys:
        t = _KEY_TO_TYPE.get(k.lower())
        if t:
            return t
    if any(w in name_lower for w in ["opq", "personality", "motivational", " mq"]):
        return "P"
    if any(w in name_lower for w in ["verify", "numerical", "verbal", "inductive",
                                      "deductive", "reasoning", "aptitude"]):
        return "A"
    if any(w in name_lower for w in ["coding", "automata", "simulation", "smart interview"]):
        return "S"
    if any(w in name_lower for w in ["sjt", "situational"]):
        return "B"
    return "K"


def _parse_duration(raw: str) -> Optional[int]:
    if not raw:
        return None
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else None


# ── Role-signal expansion ──────────────────────────────────────────────────
# These phrases appear in descriptions and signal what ROLE a test targets.
# We extract and repeat them in embed_text so semantic search can discriminate.

_TECH_SIGNALS = re.compile(
    r"\b(software|engineer|developer|programming|coding|java|python|sql|"
    r"database|data|cloud|devops|linux|network|web|frontend|backend|"
    r"full[- ]?stack|\.net|javascript|react|angular|aws|azure|"
    r"machine learning|ml|ai|algorithm|api|rest|microservice|"
    r"administration|excel|word|office|accounting|finance|sales|"
    r"customer|service|manager|leadership|hr|recruitment|"
    r"project management|agile|scrum)\b",
    re.I,
)


def _extract_role_signals(text: str) -> List[str]:
    """Pull out domain/role keywords from description for embedding enrichment."""
    return list({m.lower() for m in _TECH_SIGNALS.findall(text)})


class CatalogStore:
    def __init__(self):
        self._by_id:   Dict[str, Assessment] = {}
        self._by_name: Dict[str, Assessment] = {}
        self._all:     List[Assessment] = []

    def load(self, path: str = _CATALOG_PATH) -> None:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.warning("Catalog not found at %s", path)
            return

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        for item in raw:
            keys = item.get("keys") or item.get("test_type_keys") or []
            if isinstance(keys, str):
                keys = [keys]

            a = Assessment(
                entity_id=str(item.get("entity_id", item.get("id", ""))),
                name=item.get("name", ""),
                url=item.get("link", item.get("url", "")),
                description=item.get("description", ""),
                job_levels=item.get("job_levels", []),
                languages=item.get("languages", []),
                duration_minutes=_parse_duration(item.get("duration", "")),
                remote_testing=str(item.get("remote", "no")).lower() in ("yes", "true", "1"),
                adaptive_irt=str(item.get("adaptive", "no")).lower() in ("yes", "true", "1"),
                keys=keys,
                test_type=_infer_type(keys, item.get("name", "")),
            )
            a.embed_text = self._build_embed_text(a)

            self._by_id[a.entity_id]       = a
            self._by_name[a.name.lower()]  = a
            self._all.append(a)

        logger.info("Loaded %d assessments from catalog", len(self._all))

    @staticmethod
    def _build_embed_text(a: Assessment) -> str:
        """
        Build a rich embedding string that gives the semantic model enough signal
        to distinguish a Java test from an Excel test from a personality test.

        Strategy:
        - Name repeated twice (it's the strongest signal)
        - Full description
        - Role/domain keywords extracted from description (repeated for weight)
        - Keys (test category)
        - Job levels
        """
        role_signals = _extract_role_signals(a.description + " " + a.name)

        parts = [
            # Name twice — highest signal
            a.name,
            a.name,
            # Description — semantic content
            a.description,
            # Extracted domain keywords repeated — boosts similarity for role matches
            " ".join(role_signals) if role_signals else "",
            " ".join(role_signals) if role_signals else "",
            # Structural metadata
            "assessment type: " + ", ".join(a.keys) if a.keys else "",
            "job levels: " + ", ".join(a.job_levels) if a.job_levels else "",
        ]
        return " | ".join(p for p in parts if p)

    # ── Lookup helpers ───────────────────────────────────────────────────────

    def get_by_id(self, entity_id: str) -> Optional[Assessment]:
        return self._by_id.get(str(entity_id))

    def get_by_name(self, name: str) -> Optional[Assessment]:
        return self._by_name.get(name.lower())

    def all(self) -> List[Assessment]:
        return list(self._all)

    def filter(
        self,
        job_levels:   Optional[List[str]] = None,
        test_types:   Optional[List[str]] = None,
        max_duration: Optional[int] = None,
        remote_only:  bool = False,
    ) -> List[Assessment]:
        results = self._all
        if job_levels:
            jl_lower = {j.lower() for j in job_levels}
            results = [a for a in results if any(j.lower() in jl_lower for j in a.job_levels)]
        if test_types:
            tt_set = {t.upper() for t in test_types}
            results = [a for a in results if a.test_type in tt_set]
        if max_duration is not None:
            results = [a for a in results if
                       a.duration_minutes is None or a.duration_minutes <= max_duration]
        if remote_only:
            results = [a for a in results if a.remote_testing]
        return results

    def search_by_name(self, query: str, top_k: int = 5) -> List[Assessment]:
        q = query.lower()
        exact   = [a for a in self._all if q == a.name.lower()]
        if exact:
            return exact[:top_k]
        partial = [a for a in self._all if q in a.name.lower()]
        return partial[:top_k]

    def size(self) -> int:
        return len(self._all)