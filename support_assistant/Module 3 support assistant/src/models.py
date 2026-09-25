from typing import TypedDict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str


class QAResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: list[dict]
    response: dict