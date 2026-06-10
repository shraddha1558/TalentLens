"""
Clarifier agent:
Used when the system does not have enough information (missing slots).

It asks targeted questions instead of generating recommendations.
"""

import logging
from typing import List

from app.models.request import Message
from app.models.response import AgentResponse
from app.agent.slot_extractor import Slots

logger = logging.getLogger(__name__)


class Clarifier:
    def missing_slots(self, slots: Slots) -> List[str]:
        missing = []

        if not slots.role:
            missing.append("role")
        if not slots.seniority:
            missing.append("seniority")
        if not slots.skills:
            missing.append("skills/purpose")

        return missing

    async def ask(self, messages: List[Message], slots: Slots) -> AgentResponse:
        missing = self.missing_slots(slots)

        if not missing:
            question = "Could you provide a bit more detail about the role you're hiring for?"
        else:
            question = self._build_question(missing)

        return AgentResponse(
            reply=question,
            recommendations=[],
            end_of_conversation=False,
        )

    def _build_question(self, missing: List[str]) -> str:
        if "role" in missing:
            return "What role are you trying to assess or hire for?"
        if "seniority" in missing:
            return "What seniority level are you looking for (junior, mid, senior)?"
        if "skills/purpose" in missing:
            return "What key skills or job purpose should the assessment focus on?"
        return "Could you share more details about your hiring requirements?"