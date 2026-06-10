"""
Orchestrator — top-level agent brain.

Flow per turn:
  1. Injection guard
  2. Classify intent
  3. Extract slots
  4. Route to sub-agent

Key fixes vs original:
- Uses clarifier.needs_clarification(messages, slots) instead of missing_slots()
  so the turn-cap and seniority-inference logic is respected.
- CLARIFY intent from the classifier no longer blindly delegates to clarifier.ask();
  it first checks needs_clarification() so a role-known conversation never loops.
- Only one Clarifier instance is created per run() call.
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

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, catalog: CatalogStore, vector_store: VectorStore):
        self.catalog = catalog
        self.vector_store = vector_store
        self.hybrid = HybridSearch(catalog, vector_store)

    async def run(self, messages: List[Message]) -> AgentResponse:

        # ── 0. Injection / off-topic guard ──────────────────────────────────
        last_user_msg = self._last_user(messages)
        if _is_injection(last_user_msg):
            return AgentResponse(
                reply=(
                    "I'm here to help you find the right SHL assessments. "
                    "I can't help with that request. What role are you looking to assess?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 1. Classify intent ───────────────────────────────────────────────
        intent = await classify_intent(messages)
        logger.info("Intent: %s", intent)

        # ── 2. Extract slots ─────────────────────────────────────────────────
        slots = await SlotExtractor().extract(messages)
        logger.info("Slots: %s", slots)

        # Single clarifier instance for this request
        clarifier = Clarifier()

        # ── 3. Route ─────────────────────────────────────────────────────────

        if intent == Intent.REFUSE:
            return AgentResponse(
                reply=(
                    "I specialise only in SHL assessments and can't help with that. "
                    "What role are you trying to fill?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        if intent == Intent.CONFIRM:
            return AgentResponse(
                reply="Glad the recommendations are helpful! Feel free to reach out any time.",
                recommendations=[],
                end_of_conversation=True,
            )

        if intent == Intent.COMPARE:
            return await Comparer(self.catalog).compare(messages, slots)

        if intent == Intent.REFINE:
            return await Refiner(self.catalog, self.hybrid).refine(messages, slots)

        # For CLARIFY intent: only ask if we genuinely still need info.
        # If the classifier said CLARIFY but slots are sufficient (e.g. role is
        # known and turn cap is hit), fall through to RECOMMEND.
        if intent == Intent.CLARIFY:
            if clarifier.needs_clarification(messages, slots):
                return await clarifier.ask(messages, slots)
            # Fall through to recommend

        # RECOMMEND (or CLARIFY that resolved to sufficient context):
        # One last check — if role is truly unknown and we haven't hit the cap,
        # ask for it. Otherwise recommend with what we have.
        if clarifier.needs_clarification(messages, slots):
            logger.info("Falling back to clarification before recommend")
            return await clarifier.ask(messages, slots)

        return await Recommender(self.catalog, self.hybrid).recommend(messages, slots)

    @staticmethod
    def _last_user(messages: List[Message]) -> str:
        for m in reversed(messages):
            if m.role == "user":
                return m.content
        return ""


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