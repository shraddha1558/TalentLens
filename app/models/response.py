from typing import List, Optional
from pydantic import BaseModel, field_validator


class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str  # K=Knowledge, P=Personality, A=Ability, S=Simulation, B=Biodata


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