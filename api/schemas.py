# api/schemas.py
"""Formes des requêtes et réponses de l'API (validées par Pydantic)."""


from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    use_rag: bool = True
    top_k: int = Field(3, ge=1, le=10)
    max_new_tokens: int = Field(150, ge=1, le=512)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
