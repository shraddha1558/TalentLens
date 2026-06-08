"""
Recommender: generates a grounded, bundle-aware shortlist.

Strategy:
  1. Run hybrid search with extracted slots
  2. Enforce bundle logic (Knowledge + Personality + optional Simulation)
  3. Call LLM to write a consultant-style explanation
  4. Return structured AgentResponse
"""

import logging
from typing import List, Optional

from app.models.request import Message
from app.models.response import AgentResponse, RecommendationItem
from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.hybrid_search import HybridSearch
from app.agent.slot_extractor import Slots
from app.agent import llm

logger = logging.getLogger(__name__)

_SYSTEM_RECOMMEND = """\
You are a senior SHL solutions consultant. You have retrieved the following assessment candidates
from the SHL catalog. Write a concise, professional recommendation reply.

FORMAT:
1. One sentence acknowledging the role/context.
2. One sentence explaining your selection rationale (what the bundle covers).
3. Brief mention of next steps or a single optional follow-up question IF genuinely useful.

DO NOT:
- Invent any assessments not in the CATALOG CANDIDATES list.
- Include URLs or entity IDs in your reply text — those come from the structured data.
- Write more than 5 sentences total.
- Use bullet points or numbered lists in your reply.

ROLE CONTEXT: {role_context}
CATALOG CANDIDATES (use only these):
{catalog_summary}
"""


class Recommender:
    def __init__(self, catalog: CatalogStore, hybrid: HybridSearch):
        self.catalog = catalog
        self.hybrid = hybrid

    async def recommend(self, messages: List[Message], slots: Slots) -> AgentResponse:
        # 1. Run retrieval
        query = slots.build_search_query()
        raw_candidates = self.hybrid.search(
            query=query,
            seniority=slots.seniority,
            desired_types=slots.desired_test_types(),
            max_duration=slots.max_duration_minutes,
            remote_only=slots.remote_only,
            top_k=10,
        )

        # 2. Bundle logic
        selected = _apply_bundle_logic(raw_candidates, slots)

        if not selected:
            return AgentResponse(
                reply=(
                    "I wasn't able to find matching assessments in the SHL catalog for that query. "
                    "Could you provide more details about the role or required skills?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # 3. LLM explanation
        role_context = _describe_context(slots)
        catalog_summary = "\n".join(
            f"- {a.name} ({a.test_type}) | {a.display_duration} | {'; '.join(a.keys)}"
            for a in selected
        )
        history = [{"role": m.role, "content": m.content} for m in messages[-4:]]

        try:
            reply = await llm.call_llm(
                system=_SYSTEM_RECOMMEND.format(
                    role_context=role_context,
                    catalog_summary=catalog_summary,
                ),
                messages=history,
                temperature=0.3,
                max_tokens=300,
            )
        except Exception as e:
            logger.warning("Recommender LLM failed: %s — using fallback reply", e)
            reply = f"Based on your requirements, here are {len(selected)} assessments from the SHL catalog that I'd recommend."

        items = [
            RecommendationItem(name=a.name, url=a.url, test_type=a.test_type)
            for a in selected
        ]

        return AgentResponse(
            reply=reply.strip(),
            recommendations=items,
            end_of_conversation=False,
        )


# ── Bundle logic ─────────────────────────────────────────────────────────────

def _apply_bundle_logic(candidates: List[Assessment], slots: Slots) -> List[Assessment]:
    """
    Enforce a sensible coverage bundle:
      - Always include top K/S (knowledge / simulation) items for the role
      - Add P (personality) unless user explicitly didn't ask for it
      - Cap at 10

    This mirrors SHL's own sales approach: bundle instruments, not singles.
    """
    knowledge = [a for a in candidates if a.test_type in ("K", "S")]
    personality = [a for a in candidates if a.test_type == "P"]
    ability = [a for a in candidates if a.test_type == "A"]
    other = [a for a in candidates if a.test_type == "B"]

    result: List[Assessment] = []

    # Core: top knowledge/simulation matches (up to 5)
    result.extend(knowledge[:5])

    # Personality: add 1-2 unless user explicitly excluded
    if slots.wants_personality or not _user_excluded_type(slots, "P"):
        result.extend(personality[:2])

    # Cognitive ability: add 1 if user asked or if < 4 items so far
    if slots.wants_cognitive or len(result) < 4:
        result.extend(ability[:1])

    # Deduplicate preserving order
    seen = set()
    deduped = []
    for a in result:
        if a.entity_id not in seen:
            seen.add(a.entity_id)
            deduped.append(a)

    # Fill to at least 1, at most 10
    if not deduped:
        deduped = candidates[:10]

    return deduped[:10]


def _user_excluded_type(slots: Slots, t: str) -> bool:
    # In future, a "negative slot" could track explicit exclusions
    # For now, return False (never assumed excluded)
    return False


def _describe_context(slots: Slots) -> str:
    parts = []
    if slots.role:
        parts.append(slots.role)
    if slots.seniority:
        parts.append(f"({slots.seniority} level)")
    if slots.purpose:
        parts.append(f"for {slots.purpose}")
    return " ".join(parts) if parts else "unspecified role"