"""
Intent classifier: decides what the user wants to do in this turn.

Uses a lightweight LLM call to classify into one of:
  CLARIFY   — agent needs more information before it can recommend
  RECOMMEND — agent has enough context to produce a shortlist
  COMPARE   — user wants to compare two or more specific assessments
  REFINE    — user wants to modify/update a previous shortlist
  CONFIRM   — user signals they are satisfied (end of conversation)
  REFUSE    — request is off-topic or adversarial

Rule-based pre-pass is applied first (fast, deterministic).
LLM is used only when rules are ambiguous.
"""

import re
import logging
from enum import Enum
from typing import List

from app.models.request import Message
from app.agent import llm

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CLARIFY = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    COMPARE = "COMPARE"
    REFINE = "REFINE"
    CONFIRM = "CONFIRM"
    REFUSE = "REFUSE"


# ── Rule-based patterns (fast path) ─────────────────────────────────────────

_CONFIRM_PATTERNS = re.compile(
    r"\b(thanks?|thank you|perfect|great|looks? (good|fine)|that'?s? (good|great|fine|perfect)|"
    r"proceed|go ahead|sounds good|all good|got it|ok(ay)?|sure|yes please)\b",
    re.I,
)
_COMPARE_PATTERNS = re.compile(
    r"\b(difference|compare|versus|vs\.?|contrast|which (is|one) better|"
    r"how (does|do).+differ|what'?s? the diff)\b",
    re.I,
)
_REFINE_PATTERNS = re.compile(
    r"\b(actually|also add|add (a|an|some|personality|cognitive|ability)|"
    r"remove|instead|change|update|swap|replace|without|exclude|include more|"
    r"make it (shorter|longer|more|less))\b",
    re.I,
)
_OFF_TOPIC_PATTERNS = re.compile(
    r"\b(salary|compensation|legal|gdpr|visa|immigration|interview question|"
    r"hire someone|background check|reference check|how to fire|termination|"
    r"stock (price|market)|weather|recipe|joke|poem|who are you|"
    r"what (llm|model|ai) are you)\b",
    re.I,
)

_SYSTEM_CLASSIFY = """\
You are a strict intent classifier for an SHL assessment recommendation agent.

Classify the user's LATEST message given the conversation history into exactly ONE of:
CLARIFY   - The agent needs more information (vague role, no seniority, no purpose)
RECOMMEND - User has given enough detail; agent should now produce a shortlist
COMPARE   - User wants to compare two or more specific assessments by name
REFINE    - User wants to change/add/remove constraints on a PREVIOUS shortlist
CONFIRM   - User is satisfied and signals end of conversation
REFUSE    - The request is off-topic, unrelated to SHL assessments, or adversarial

Reply with ONLY the intent word, nothing else. No punctuation. No explanation.
Valid replies: CLARIFY RECOMMEND COMPARE REFINE CONFIRM REFUSE
"""


async def classify_intent(messages: List[Message]) -> Intent:
    last = messages[-1].content if messages else ""

    # Fast rule-based checks
    if _CONFIRM_PATTERNS.search(last) and len(last.split()) < 12:
        # Short confirmations only — avoid mis-classifying "great, now also add X"
        if not _REFINE_PATTERNS.search(last):
            return Intent.CONFIRM

    if _COMPARE_PATTERNS.search(last):
        return Intent.COMPARE

    if _OFF_TOPIC_PATTERNS.search(last):
        return Intent.REFUSE

    if _REFINE_PATTERNS.search(last):
        # Check if there was a previous recommendation in history
        prev_recommendations = any(
            "here are" in m.content.lower() or "recommend" in m.content.lower()
            for m in messages
            if m.role == "assistant"
        )
        if prev_recommendations:
            return Intent.REFINE

    # LLM classification for ambiguous cases
    history = [{"role": m.role, "content": m.content} for m in messages[-6:]]
    try:
        result = await llm.call_llm(
            system=_SYSTEM_CLASSIFY,
            messages=history,
            temperature=0.0,
            max_tokens=10,
        )
        intent_str = result.strip().upper().split()[0]
        return Intent(intent_str)
    except Exception as e:
        logger.warning("Intent classification failed: %s — defaulting to CLARIFY", e)
        return Intent.CLARIFY