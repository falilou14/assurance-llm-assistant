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


class AgentAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    max_turns: int = Field(5, ge=1, le=10)


class AgentToolCall(BaseModel):
    outil: str
    arguments: dict
    resultat: dict


class AgentAskResponse(BaseModel):
    answer: str
    trace: list[AgentToolCall]
    nb_turns: int
    latency_ms: float
