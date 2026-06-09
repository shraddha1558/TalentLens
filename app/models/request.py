from typing import List, Literal
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be blank")
        return v.strip()


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def must_start_with_user(cls, v):
        if v and v[0].role != "user":
            raise ValueError("First message must be from the user")
        return v