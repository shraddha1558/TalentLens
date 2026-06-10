"""
Clarifier agent.

Rules:
- Role is the only hard requirement. Without it, always ask.
- Seniority is soft: ask once, then move on regardless of answer.
- After MAX_CLARIFICATION_TURNS assistant questions, stop asking and recommend.
- Detects seniority from free text so it never asks for what's already stated.
- missing_slots() is kept for orchestrator compatibility.
"""

import logging
import re
from typing import List, Optional

from app.models.request import Message
from app.models.response import AgentResponse
from app.agent.slot_extractor import Slots

logger = logging.getLogger(__name__)

MAX_CLARIFICATION_TURNS = 2  # max assistant questions before we just recommend

# Free-text seniority signals
_SENIORITY_HINTS = {
    "junior":  ["junior", "entry", "entry-level", "fresher", "graduate", "new grad"],
    "mid":     ["mid", "mid-level", "intermediate", "2-5 year", "3-5 year"],
    "senior":  ["senior", "sr.", "sr ", "lead", "staff", "principal", "architect", "expert"],
    "manager": ["manager", "director", "head of", "vp ", "executive", "c-level"],
}


def _infer_seniority(messages: List[Message]) -> Optional[str]:
    """Scan full conversation for seniority keywords."""
    text = " ".join(m.content.lower() for m in messages)
    for level, hints in _SENIORITY_HINTS.items():
        if any(h in text for h in hints):
            return level
    match = re.search(r"(\d+)\s*\+?\s*years?", text)
    if match:
        y = int(match.group(1))
        return "junior" if y <= 2 else ("mid" if y <= 5 else "senior")
    return None


def _assistant_question_count(messages: List[Message]) -> int:
    """Count how many turns the assistant has already asked a question."""
    return sum(1 for m in messages if m.role == "assistant" and "?" in m.content)


def _already_asked_seniority(messages: List[Message]) -> bool:
    markers = ["seniority", "experience level", "junior", "senior", "mid-level", "years of experience"]
    return any(
        m.role == "assistant" and any(k in m.content.lower() for k in markers)
        for m in messages
    )


class Clarifier:

    def missing_slots(self, slots: Slots) -> List[str]:
        """
        Returns list of slot names still needed.
        Kept for backward-compat with orchestrator.
        Only 'role' is truly required.
        """
        missing = []
        if not slots.role:
            missing.append("role")
        return missing

    def needs_clarification(self, messages: List[Message], slots: Slots) -> bool:
        """
        True only when we genuinely need more info AND haven't exceeded turn limit.
        """
        if _assistant_question_count(messages) >= MAX_CLARIFICATION_TURNS:
            return False

        if not slots.role:
            return True

        if not slots.seniority:
            inferred = _infer_seniority(messages)
            if inferred:
                slots.seniority = inferred
                return False
            if not _already_asked_seniority(messages):
                return True

        return False

    async def ask(self, messages: List[Message], slots: Slots) -> AgentResponse:
        if _assistant_question_count(messages) >= MAX_CLARIFICATION_TURNS:
            return AgentResponse(
                reply="Let me find the best assessments based on what you've shared.",
                recommendations=[],
                end_of_conversation=False,
            )

        question = self._pick_question(messages, slots)
        logger.info("Clarifier asking: %s", question)
        return AgentResponse(reply=question, recommendations=[], end_of_conversation=False)

    def _pick_question(self, messages: List[Message], slots: Slots) -> str:
        if not slots.role:
            return "What role are you hiring or assessing for?"

        if not slots.seniority and not _already_asked_seniority(messages):
            role_str = f" for this {slots.role} role" if slots.role else ""
            return (
                f"What experience level are you targeting{role_str}? "
                "(e.g. junior, mid-level, senior — or skip and I'll proceed.)"
            )

        return "Any other constraints I should know about, like assessment duration or remote delivery?"