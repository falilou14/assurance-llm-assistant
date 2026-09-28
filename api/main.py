# api/main.py
"""
API de l'assistant assurance -- expose le RAG + LoRA derrière HTTP.

Le modèle est chargé UNE SEULE FOIS au démarrage (lifespan), pas à
chaque requête -- c'est l'équivalent serveur du st.cache_resource utilisé
côté Streamlit.
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from api.deps import LLMClientProtocol, get_llm_client
from api.schemas import AskRequest, AskResponse, HealthResponse, ReadyResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("APP_ENV") == "test":
        # En test, on ne charge jamais le vrai modèle : tests/test_api.py
        # fournit un faux client via app.dependency_overrides.
        app.state.llm_client = None
    else:
        from src.llm_client import LocalLLMClient  # import lourd -- seulement ici

        app.state.llm_client = LocalLLMClient()
    yield
    app.state.llm_client = None


app = FastAPI(title="Assurance LLM API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Le processus est-il vivant ? Ne dépend jamais du modèle (liveness)."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """Le modèle est-il chargé et prêt à servir ? (readiness)"""
    return ReadyResponse(ready=request.app.state.llm_client is not None)


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    client: LLMClientProtocol = Depends(get_llm_client),
) -> AskResponse:
    # def, pas async def : la génération est un calcul CPU bloquant de
    # plusieurs secondes -- dans une coroutine async, elle figerait tout
    # le serveur. FastAPI exécute les endpoints "def" dans un pool de
    # threads, ce qui garde le serveur réactif pour les autres requêtes.
    t0 = time.perf_counter()
    answer, sources = client.generate(
        question=payload.question,
        use_rag=payload.use_rag,
        top_k=payload.top_k,
        max_new_tokens=payload.max_new_tokens,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return AskResponse(answer=answer, sources=sources, latency_ms=round(latency_ms, 1))
