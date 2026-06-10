"""
Slot Extractor: converts chat messages into structured hiring requirements.

This is used by:
  - Recommender
  - Comparer
  - Refiner

It extracts "slots" like:
  - role
  - seniority
  - skills
  - test preferences
  - constraints (duration, remote, etc.)

Implementation: LLM-based structured extraction with safe fallback.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.models.request import Message
from app.agent import llm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Slots data model
# ─────────────────────────────────────────────────────────────

@dataclass
class Slots:
    role: Optional[str] = None
    seniority: Optional[str] = None
    purpose: Optional[str] = None

    skills: List[str] = field(default_factory=list)

    desired_test_types: List[str] = field(default_factory=list)

    max_duration_minutes: Optional[int] = None
    remote_only: bool = False

    wants_personality: bool = True
    wants_cognitive: bool = True

    def build_search_query(self) -> str:
        parts = []

        if self.role:
            parts.append(self.role)

        if self.seniority:
            parts.append(self.seniority)

        if self.skills:
            parts.extend(self.skills)

        if self.purpose:
            parts.append(self.purpose)

        return " ".join(parts) if parts else "assessment"


# ─────────────────────────────────────────────────────────────
# Slot extractor
# ─────────────────────────────────────────────────────────────

_SYSTEM_SLOTS = """
You are a strict information extraction system for an SHL assessment recommender.

Extract structured data from the conversation.

Return ONLY valid JSON in this format:

{
  "role": string or null,
  "seniority": string or null,
  "purpose": string or null,
  "skills": [string],
  "desired_test_types": [string],
  "max_duration_minutes": number or null,
  "remote_only": boolean,
  "wants_personality": boolean,
  "wants_cognitive": boolean
}

Rules:
- If unknown, use null or empty list.
- Do NOT include explanations.
- Output ONLY JSON.
"""


class SlotExtractor:
    async def extract(self, messages: List[Message]) -> Slots:
        """
        Extract structured slots from full conversation history.
        """

        history = [
            {"role": m.role, "content": m.content}
            for m in messages[-8:]
        ]

        try:
            raw = await llm.call_llm(
                system=_SYSTEM_SLOTS,
                messages=history,
                temperature=0.0,
                max_tokens=300,
                json_mode=True,
            )

            data = json.loads(raw)

            logger.info("EXTRACTED SLOTS: %s", data)

            return Slots(
                role=data.get("role"),
                seniority=data.get("seniority"),
                purpose=data.get("purpose"),
                skills=data.get("skills") or [],
                desired_test_types=data.get("desired_test_types") or [],
                max_duration_minutes=data.get("max_duration_minutes"),
                remote_only=data.get("remote_only", False),
                wants_personality=data.get("wants_personality", True),
                wants_cognitive=data.get("wants_cognitive", True),
            )

        except Exception as e:
            logger.warning(
                "Slot extraction failed: %s — using fallback",
                e
            )

            text = " ".join(
                m.content
                for m in messages
                if m.role == "user"
            )

            return Slots(
                role=text[:80],
                skills=[],
                desired_test_types=[],
                remote_only=False,
                wants_personality=True,
                wants_cognitive=True,
            )