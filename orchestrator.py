"""
Orchestrator: the top-level agent brain.

Flow per turn:
  1. Classify intent (CLARIFY / RECOMMEND / COMPARE / REFINE / CONFIRM / REFUSE)
  2. Extract slots from full conversation history
  3. Route to the appropriate sub-agent
  4. Return AgentResponse (schema-compliant)

All LLM calls go through a thin wrapper in agent/llm.py so swapping
providers only requires changing one file.
"""

import logging
from typing import List

from app.models.request import Message
from app.models.response import AgentResponse
from app.retrieval.catalog_store import CatalogStore
from app.retrieval.vector_store import VectorStore
from app.retrieval.hybrid_search import HybridSearch
from app.agent.classifier import classify_intent, Intent
from app.agent.slot_extractor import SlotExtractor
from app.agent.clarifier import Clarifier
from app.agent.recommender import Recommender
from app.agent.comparer import Comparer
from app.agent.refiner import Refiner
from app.agent import llm

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, catalog: CatalogStore, vector_store: VectorStore):
        self.catalog = catalog
        self.vector_store = vector_store
        self.hybrid = HybridSearch(catalog, vector_store)

    async def run(self, messages: List[Message]) -> AgentResponse:
        # ── 0. Prompt-injection / off-topic guard ───────────────────────────
        last_user_msg = self._last_user(messages)
        if _is_injection(last_user_msg):
            return AgentResponse(
                reply=(
                    "I'm here to help you find the right SHL assessments for your hiring needs. "
                    "I can't help with that request. What role are you looking to assess?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 1. Classify intent ──────────────────────────────────────────────
        intent = await classify_intent(messages)
        logger.info("Intent: %s", intent)

        # ── 2. Extract slots from full history ──────────────────────────────
        slots = await SlotExtractor().extract(messages)
        logger.info("Slots: %s", slots)

        # ── 3. Route ────────────────────────────────────────────────────────
        if intent == Intent.REFUSE:
            return AgentResponse(
                reply=(
                    "I specialise only in recommending SHL assessments. I'm not able to help "
                    "with general hiring advice, legal questions, or topics outside the SHL catalog. "
                    "What role are you trying to fill?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if intent == Intent.CONFIRM:
            return AgentResponse(
                reply=(
                    "Great! I'm glad the recommendations are helpful. "
                    "Feel free to reach out if you need further assessment advice."
                ),
                recommendations=[],
                end_of_conversation=True,
            )

        if intent == Intent.COMPARE:
            comparer = Comparer(self.catalog)
            return await comparer.compare(messages, slots)

        if intent == Intent.CLARIFY:
            clarifier = Clarifier()
            return await clarifier.ask(messages, slots)

        if intent == Intent.REFINE:
            refiner = Refiner(self.catalog, self.hybrid)
            return await refiner.refine(messages, slots)

        # Default: RECOMMEND
        # If slots are insufficient, fall back to clarification
        clarifier = Clarifier()
        missing = clarifier.missing_slots(slots)
        if missing:
            logger.info("Falling back to clarification — missing: %s", missing)
            return await clarifier.ask(messages, slots)

        recommender = Recommender(self.catalog, self.hybrid)
        return await recommender.recommend(messages, slots)

    @staticmethod
    def _last_user(messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""


# Very lightweight injection guard — LLM classifier handles edge cases
_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your",
    "you are now",
    "act as",
    "pretend you",
    "forget your",
    "new persona",
    "system prompt",
    "jailbreak",
    "dan mode",
]


def _is_injection(text: str) -> bool:
    t = text.lower()
    return any(phrase in t for phrase in _INJECTION_PHRASES)