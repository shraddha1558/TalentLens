"""
Refiner Agent

Handles updates to previously provided requirements.

Examples:
- "Actually add personality tests."
- "Make it suitable for graduates."
- "Remove cognitive tests."
- "Only assessments under 30 minutes."
"""

import logging
from typing import List

from app.models.request import Message
from app.models.response import AgentResponse, RecommendationItem
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)


class Refiner:
    def __init__(
        self,
        catalog: CatalogStore,
        hybrid: HybridSearch,
    ):
        self.catalog = catalog
        self.hybrid = hybrid

    async def refine(
        self,
        messages: List[Message],
        slots: dict,
    ) -> AgentResponse:

        try:
            query = self._build_query(slots)

            logger.info("Refinement query: %s", query)

            results = self.hybrid.search(
                query=query,
                top_k=10,
            )

            recommendations = []

            for item in results[:10]:

                # Handle dict results
                if isinstance(item, dict):
                    recommendations.append(
                        RecommendationItem(
                            name=item.get("name", ""),
                            url=item.get("url", ""),
                            test_type=item.get("test_type", "K"),
                        )
                    )

                # Handle object results
                else:
                    recommendations.append(
                        RecommendationItem(
                            name=getattr(item, "name", ""),
                            url=getattr(item, "url", ""),
                            test_type=getattr(item, "test_type", "K"),
                        )
                    )

            if not recommendations:
                return AgentResponse(
                    reply=(
                        "I couldn't find any SHL assessments matching the "
                        "updated requirements. Could you provide a bit more detail?"
                    ),
                    recommendations=[],
                    end_of_conversation=False,
                )

            return AgentResponse(
                reply=self._build_reply(slots, len(recommendations)),
                recommendations=recommendations,
                end_of_conversation=False,
            )

        except Exception as e:
            logger.exception("Refinement failed: %s", str(e))

            return AgentResponse(
                reply=(
                    "I wasn't able to update the shortlist right now. "
                    "Could you rephrase the change you'd like to make?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

    def _build_query(self, slots: dict) -> str:
        parts = []

        role = slots.get("role")
        if role:
            parts.append(role)

        seniority = slots.get("seniority")
        if seniority:
            parts.append(seniority)

        skills = slots.get("skills", [])
        if skills:
            parts.extend(skills)

        assessment_types = slots.get("assessment_types", [])
        if assessment_types:
            parts.extend(assessment_types)

        traits = slots.get("traits", [])
        if traits:
            parts.extend(traits)

        return " ".join(parts)

    def _build_reply(self, slots: dict, count: int) -> str:
        role = slots.get("role")

        if role:
            return (
                f"I've updated the shortlist based on your revised "
                f"requirements for a {role}. Here are {count} relevant "
                f"SHL assessments."
            )

        return (
            f"I've updated the shortlist using your latest requirements "
            f"and found {count} relevant SHL assessments."
        )