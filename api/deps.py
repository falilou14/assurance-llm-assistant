# api/deps.py
"""
Fournit le client LLM aux endpoints (injection de dépendances FastAPI).

Ce module ne doit JAMAIS importer torch/transformers/faiss -- c'est ce
qui permet de tester l'API sans les bibliothèques lourdes du ML (voir
tests/test_api.py, qui remplace get_llm_client par un faux client).
Le vrai client (LocalLLMClient, src/llm_client.py) n'est importé qu'au
démarrage réel du serveur, dans le lifespan de api/main.py.
"""

from typing import Protocol, runtime_checkable

from fastapi import HTTPException, Request


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Contrat que doit respecter tout backend de génération (réel ou factice)."""

    def generate(
        self, question: str, use_rag: bool, top_k: int, max_new_tokens: int
    ) -> tuple[str, list[str]]:
        ...


def get_llm_client(request: Request) -> LLMClientProtocol:
    """
    Renvoie le client stocké sur app.state au démarrage.
    Si le modèle n'a pas fini de charger (ou n'a pas pu), 503 plutôt
    qu'un plantage -- c'est ce que /ready sert à vérifier en amont.
    """
    client = getattr(request.app.state, "llm_client", None)
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé -- réessaie dans quelques instants.",
        )
    return client
