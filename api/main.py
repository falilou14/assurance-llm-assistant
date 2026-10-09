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

from fastapi import Depends, FastAPI, HTTPException, Request

from api.deps import AgentRunnerProtocol, LLMClientProtocol, get_agent_runner, get_llm_client
from api.schemas import (
    AgentAskRequest,
    AgentAskResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    ReadyResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("APP_ENV") == "test":
        # En test, on ne charge jamais le vrai modèle ni le vrai agent :
        # tests/test_api.py fournit des faux clients via app.dependency_overrides.
        app.state.llm_client = None
        app.state.agent_runner = None
    else:
        from src.llm_client import LocalLLMClient  # import lourd -- seulement ici

        app.state.llm_client = LocalLLMClient()

        # Contrairement à LocalLLMClient, l'agent n'a rien à charger en RAM
        # une seule fois (pas de poids) -- run_agent() crée sa propre
        # connexion Gemini à chaque appel. On se contente de brancher la
        # fonction ; la vraie vérification (GEMINI_API_KEY présente) n'a
        # lieu qu'au premier appel, voir /agent/ask ci-dessous.
        from src.agent.loop import run_agent

        app.state.agent_runner = run_agent
    yield
    app.state.llm_client = None
    app.state.agent_runner = None


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


@app.post("/agent/ask", response_model=AgentAskResponse)
def agent_ask(
    payload: AgentAskRequest,
    runner: AgentRunnerProtocol = Depends(get_agent_runner),
) -> AgentAskResponse:
    # Même raisonnement que /ask : def, pas async def -- run_agent() fait
    # des appels réseau bloquants (API Gemini), potentiellement plusieurs
    # tours ; FastAPI l'exécute dans un pool de threads.
    t0 = time.perf_counter()
    try:
        result = runner(payload.question, max_turns=payload.max_turns)
    except RuntimeError as exc:
        # Cas concret : GEMINI_API_KEY absente -- levé par GeminiAgentClient,
        # jamais laissé remonter comme un plantage brut au client HTTP.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    latency_ms = (time.perf_counter() - t0) * 1000
    return AgentAskResponse(
        answer=result["reponse"],
        trace=result["trace"],
        nb_turns=result["nb_tours"],
        latency_ms=round(latency_ms, 1),
    )
