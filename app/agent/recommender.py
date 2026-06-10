"""
Recommender: generates a grounded, bundle-aware shortlist.

Strategy:
  1. Hybrid search → List[Tuple[Assessment, float]]
  2. Relevance gate — if best score is too low, catalog has no real match
  3. Bundle logic (Knowledge + Personality + Ability)
  4. LLM consultant-style reply
  5. Return AgentResponse
"""

import logging
from typing import List, Tuple

from app.models.request import Message
from app.models.response import AgentResponse, RecommendationItem
from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.hybrid_search import HybridSearch
from app.agent.slot_extractor import Slots
from app.agent import llm

logger = logging.getLogger(__name__)

# Hybrid score threshold. Below this we consider the catalog to have no real match.
# The hybrid score is a weighted sum in [0,1]. 0.30 is a safe starting point;
# tune by watching the "Top score" log line.
_RELEVANCE_THRESHOLD = 0.30

_SYSTEM_RECOMMEND = """\
You are a senior SHL solutions consultant. You have retrieved the following assessment
candidates from the SHL catalog. Write a concise, professional recommendation reply.

FORMAT:
1. One sentence acknowledging the role/context.
2. One sentence explaining your selection rationale (what the bundle covers).
3. Optional: one short follow-up question only if genuinely useful.

DO NOT:
- Invent assessments not in the CATALOG CANDIDATES list.
- Include URLs or entity IDs in your reply text.
- Write more than 5 sentences total.
- Use bullet points or numbered lists.

ROLE CONTEXT: {role_context}
CATALOG CANDIDATES (use only these):
{catalog_summary}
"""


class Recommender:
    def __init__(self, catalog: CatalogStore, hybrid: HybridSearch):
        self.catalog = catalog
        self.hybrid = hybrid

    async def recommend(self, messages: List[Message], slots: Slots) -> AgentResponse:

        # 1. Retrieval
        query = slots.build_search_query()
        results: List[Tuple[Assessment, float]] = self.hybrid.search(
            query=query,
            seniority=slots.seniority,
            desired_types=slots.desired_test_types,
            max_duration=slots.max_duration_minutes,
            remote_only=slots.remote_only,
            top_k=20,
        )

        if not results:
            return self._no_match_response(slots)

        # 2. Relevance gate
        best_score = results[0][1]
        logger.info("Top hybrid score for '%s': %.3f", query, best_score)

        if best_score < _RELEVANCE_THRESHOLD:
            return self._no_match_response(slots)

        relevant = [a for a, score in results if score >= _RELEVANCE_THRESHOLD]
        if not relevant:
            return self._no_match_response(slots)

        # 3. Bundle
        selected = _apply_bundle_logic(relevant, slots)

        # 4. LLM reply
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
            logger.warning("Recommender LLM failed: %s", e)
            reply = (
                f"Based on your requirements, here are {len(selected)} SHL assessments "
                "that fit this role."
            )

        return AgentResponse(
            reply=reply.strip(),
            recommendations=[
                RecommendationItem(name=a.name, url=a.url, test_type=a.test_type)
                for a in selected
            ],
            end_of_conversation=False,
        )

    def _no_match_response(self, slots: Slots) -> AgentResponse:
        role_str = f"'{slots.role}'" if slots.role else "this role"
        return AgentResponse(
            reply=(
                f"I wasn't able to find SHL assessments that closely match {role_str}. "
                "The SHL catalog covers business, technology, and professional roles — "
                "it may not include every specialisation. "
                "If you'd like, I can search using a related role or broader skill area."
            ),
            recommendations=[],
            end_of_conversation=False,
        )


# ── Bundle logic ──────────────────────────────────────────────────────────────

def _apply_bundle_logic(candidates: List[Assessment], slots: Slots) -> List[Assessment]:
    knowledge   = [a for a in candidates if a.test_type in ("K", "S")]
    personality = [a for a in candidates if a.test_type == "P"]
    ability     = [a for a in candidates if a.test_type == "A"]

    result: List[Assessment] = []
    result.extend(knowledge[:5])

    if slots.wants_personality:
        result.extend(personality[:2])

    if slots.wants_cognitive or len(result) < 4:
        result.extend(ability[:1])

    seen, deduped = set(), []
    for a in result:
        if a.entity_id not in seen:
            seen.add(a.entity_id)
            deduped.append(a)

    if not deduped:
        deduped = candidates[:10]

    return deduped[:10]


def _describe_context(slots: Slots) -> str:
    parts = []
    if slots.role:
        parts.append(slots.role)
    if slots.seniority:
        parts.append(f"({slots.seniority} level)")
    if slots.purpose:
        parts.append(f"for {slots.purpose}")
    return " ".join(parts) if parts else "unspecified role"