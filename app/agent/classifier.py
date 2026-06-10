"""
Intent classifier.

Fast rule-based pre-pass first. LLM only for genuinely ambiguous cases.

Key fixes vs original:
- Short answers ("No, that's it.", "Senior", "That's all") are treated as
  RECOMMEND when there is already hiring context in the history — never CLARIFY.
- The LLM system prompt explicitly instructs: if role is known, classify RECOMMEND.
- Dismissive answers ("no", "nope", "that's it", "no that's it") are recognised
  as "user has given all they're going to give → recommend now".
"""

import re
import logging
from enum import Enum
from typing import List

from app.models.request import Message
from app.agent import llm

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CLARIFY   = "CLARIFY"
    RECOMMEND = "RECOMMEND"
    COMPARE   = "COMPARE"
    REFINE    = "REFINE"
    CONFIRM   = "CONFIRM"
    REFUSE    = "REFUSE"


# ── Rule patterns ────────────────────────────────────────────────────────────

_CONFIRM_RE = re.compile(
    r"^\s*(thanks?|thank you|perfect|great|looks? (good|fine)|that'?s? (good|great|fine|perfect)"
    r"|proceed|go ahead|sounds good|all good|ok(ay)?|sure|yes please)\s*[.!]?\s*$",
    re.I,
)

_COMPARE_RE = re.compile(
    r"\b(difference|compare|versus|vs\.?|contrast|which (is|one) better"
    r"|how (does|do).+differ|what'?s? the diff)\b",
    re.I,
)

_REFINE_RE = re.compile(
    r"\b(actually|also add|add (a|an|some|personality|cognitive|ability)"
    r"|remove|instead|change|update|swap|replace|without|exclude"
    r"|include more|make it (shorter|longer|more|less))\b",
    re.I,
)

_OFF_TOPIC_RE = re.compile(
    r"\b(salary|compensation|legal|gdpr|visa|immigration|interview question"
    r"|background check|reference check|how to fire|termination"
    r"|stock (price|market)|weather|recipe|joke|poem"
    r"|who are you|what (llm|model|ai) are you)\b",
    re.I,
)

# Answers that mean "I have nothing more to add, just recommend"
_DISMISSIVE_RE = re.compile(
    r"^\s*(no[,.]?|nope|nah|that'?s? it\.?|no that'?s? it\.?"
    r"|nothing (else|more)|that'?s? all\.?|just (that|proceed)"
    r"|skip|no preference|doesn'?t matter|no more|done)\s*[.!]?\s*$",
    re.I,
)

_HIRING_CONTEXT_WORDS = [
    "hiring", "engineer", "developer", "manager", "analyst", "designer",
    "assessment", "candidate", "role", "job", "position", "recruit",
    "test", "evaluate",
]


def _has_hiring_context(messages: List[Message]) -> bool:
    return any(
        any(w in m.content.lower() for w in _HIRING_CONTEXT_WORDS)
        for m in messages[:-1]   # exclude current message
    )


def _role_is_known(messages: List[Message]) -> bool:
    """True if any earlier user message mentions a concrete role/job."""
    role_hints = [
        "engineer", "developer", "manager", "analyst", "designer",
        "architect", "lead", "director", "recruiter", "sales", "support",
        "java", "python", "rust", "react", "data", "ml", "devops",
    ]
    for m in messages[:-1]:
        if m.role == "user":
            lower = m.content.lower()
            if any(h in lower for h in role_hints):
                return True
    return False


_SYSTEM_CLASSIFY = """
You are an intent classifier for an SHL assessment recommendation agent.

Read the ENTIRE conversation, not just the last message.

Intent definitions:

RECOMMEND  — The agent has enough context to suggest assessments.
             Use this when:
             - A role has been mentioned (even if seniority is unknown).
             - The user says "no", "that's it", "nothing else", "skip", or
               any dismissive short answer after a role was already given.
             - The user answered a clarifying question (even with a short answer).

CLARIFY    — Role is completely unknown. Agent must ask for it.
             Do NOT use CLARIFY just because seniority is missing.

COMPARE    — User wants to compare specific assessments.

REFINE     — User wants to modify a previous shortlist.

CONFIRM    — User is satisfied, conversation is ending.

REFUSE     — Off-topic or adversarial.

Rules:
- SHORT answers like "Senior", "No", "That's it", "Mid" after a role was
  mentioned → RECOMMEND, not CLARIFY.
- Only output ONE word.
"""


async def classify_intent(messages: List[Message]) -> Intent:
    last = messages[-1].content.strip() if messages else ""

    # Confirm: very short positive signal with no refinement
    if _CONFIRM_RE.match(last) and not _REFINE_RE.search(last):
        return Intent.CONFIRM

    # Compare
    if _COMPARE_RE.search(last):
        return Intent.COMPARE

    # Off-topic
    if _OFF_TOPIC_RE.search(last):
        return Intent.REFUSE

    # Refine: only when there's a prior recommendation
    if _REFINE_RE.search(last):
        has_prior_rec = any(
            m.role == "assistant" and (
                "here are" in m.content.lower() or "recommend" in m.content.lower()
            )
            for m in messages
        )
        if has_prior_rec:
            return Intent.REFINE

    # Dismissive answer ("no", "that's it", "skip") after role is known → RECOMMEND
    if _DISMISSIVE_RE.match(last) and _role_is_known(messages):
        return Intent.RECOMMEND

    # Short contextual answer after hiring context → RECOMMEND
    short_answers = {
        "yes", "no", "nope", "senior", "junior", "mid", "mid-level",
        "that's it", "thats it", "nothing else", "no more", "proceed",
    }
    if last.lower() in short_answers and _has_hiring_context(messages):
        return Intent.RECOMMEND

    # LLM fallback for everything else
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
        logger.warning("Intent classification failed: %s — defaulting to RECOMMEND", e)
        # Default to RECOMMEND (not CLARIFY) when LLM fails and role is known
        return Intent.RECOMMEND if _role_is_known(messages) else Intent.CLARIFY