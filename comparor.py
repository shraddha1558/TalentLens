"""
Comparer: handles "difference between X and Y" queries.

Retrieves both assessments from the catalog and generates a
structured, grounded comparison. Never uses model prior knowledge.
"""

import logging
import re
from typing import List, Optional, Tuple

from app.models.request import Message
from app.models.response import AgentResponse, RecommendationItem
from app.models.assessment import Assessment
from app.retrieval.catalog_store import CatalogStore
from app.agent.slot_extractor import Slots
from app.agent import llm

logger = logging.getLogger(__name__)

_SYSTEM_COMPARE = """\
You are a senior SHL solutions consultant. Compare the two assessments below
using ONLY the provided catalog data. Do not use your own knowledge.

Write a clear, structured comparison in plain prose (3-5 sentences).
Mention key differences in: purpose, test type, job levels, duration, and when to use each.

ASSESSMENT A:
{a_info}

ASSESSMENT B:
{b_info}

End with a one-sentence recommendation on when to choose A vs B.
"""


class Comparer:
    def __init__(self, catalog: CatalogStore):
        self.catalog = catalog

    async def compare(self, messages: List[Message], slots: Slots) -> AgentResponse:
        # Extract the two assessment names from the last user message
        last = messages[-1].content
        a_name, b_name = self._extract_names(last, messages)

        a = self._find(a_name)
        b = self._find(b_name)

        if not a and not b:
            return AgentResponse(
                reply=(
                    "I couldn't find those assessments in the SHL catalog. "
                    "Could you double-check the names? "
                    "For example: 'Compare OPQ32r and Verify Numerical Reasoning'."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if not a:
            return AgentResponse(
                reply=f"I found '{b.name}' but couldn't locate '{a_name}' in the catalog. Could you clarify the name?",
                recommendations=[b_rec(b)] if b else [],
                end_of_conversation=False,
            )

        if not b:
            return AgentResponse(
                reply=f"I found '{a.name}' but couldn't locate '{b_name}' in the catalog. Could you clarify the name?",
                recommendations=[b_rec(a)] if a else [],
                end_of_conversation=False,
            )

        # Both found — generate comparison
        a_info = _format_assessment(a)
        b_info = _format_assessment(b)

        history = [{"role": m.role, "content": m.content} for m in messages[-4:]]

        try:
            reply = await llm.call_llm(
                system=_SYSTEM_COMPARE.format(a_info=a_info, b_info=b_info),
                messages=history,
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as e:
            logger.warning("Comparer LLM failed: %s", e)
            reply = f"Here is a side-by-side look at {a.name} and {b.name} based on the SHL catalog."

        items = [b_rec(a), b_rec(b)]
        return AgentResponse(
            reply=reply.strip(),
            recommendations=items,
            end_of_conversation=False,
        )

    def _find(self, name: Optional[str]) -> Optional[Assessment]:
        if not name:
            return None
        # Exact match first
        a = self.catalog.get_by_name(name)
        if a:
            return a
        # Fuzzy substring
        results = self.catalog.search_by_name(name, top_k=1)
        return results[0] if results else None

    def _extract_names(self, text: str, messages: List[Message]) -> Tuple[Optional[str], Optional[str]]:
        """
        Try to extract two assessment names from the comparison query.
        Uses multiple heuristics in priority order.
        """
        # Pattern: "X and Y", "X vs Y", "X versus Y", "between X and Y"
        patterns = [
            r"between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
            r"compare\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            r"(.+?)\s+and\s+(.+?)(?:\?|$)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                return m.group(1).strip(), m.group(2).strip()

        # Fallback: look for capitalised tokens (likely product names)
        tokens = re.findall(r"[A-Z][A-Za-z0-9\s\-]+", text)
        if len(tokens) >= 2:
            return tokens[0].strip(), tokens[1].strip()

        return None, None


def b_rec(a: Assessment) -> RecommendationItem:
    return RecommendationItem(name=a.name, url=a.url, test_type=a.test_type)


def _format_assessment(a: Assessment) -> str:
    return (
        f"Name: {a.name}\n"
        f"Type: {a.test_type} | Keys: {', '.join(a.keys)}\n"
        f"Description: {a.description}\n"
        f"Job levels: {', '.join(a.job_levels) or 'N/A'}\n"
        f"Duration: {a.display_duration}\n"
        f"Remote: {'Yes' if a.remote_testing else 'No'}\n"
        f"Adaptive: {'Yes' if a.adaptive_irt else 'No'}\n"
        f"URL: {a.url}"
    )