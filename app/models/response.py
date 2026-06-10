from typing import List, Optional
from pydantic import BaseModel, field_validator


class RecommendationItem(BaseModel):
    name: str
    url: Optional[str] = None
    entity_id: Optional[str] = None
    test_type: Optional[str] = None  # K, P, A, S, B


class AgentResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem] = []
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def cap_recommendations(cls, v):
        if len(v) > 10:
            raise ValueError("recommendations must not exceed 10 items")
        return v