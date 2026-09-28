# tests/test_api.py
"""
Tests de l'API -- n'importent JAMAIS src.llm_client, donc jamais torch.
Le LLM est remplacé par un faux client (FakeLLMClient) via
app.dependency_overrides : c'est ce qui permet à ces tests de tourner
en moins d'une seconde, sur n'importe quelle machine, sans télécharger
de modèle.
"""

import os

os.environ.setdefault("APP_ENV", "test")  # avant l'import de api.main

import pytest
from fastapi.testclient import TestClient

from api.deps import get_llm_client
from api.main import app


class FakeLLMClient:
    """Respecte LLMClientProtocol (api/deps.py) sans aucune dépendance lourde."""

    def generate(self, question, use_rag, top_k, max_new_tokens):
        sources = ["Contexte factice n°1"] if use_rag else []
        return f"Réponse factice à : {question}", sources


@pytest.fixture()
def client():
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_is_false_in_test_env(client):
    # Le lifespan ne charge jamais le vrai modèle en mode test -- /ready
    # reflète cet état réel, indépendamment du faux client injecté pour /ask.
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"ready": False}


def test_ask_returns_answer_with_fake_client(client):
    resp = client.post("/ask", json={"question": "Comment déclarer un sinistre ?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "Comment déclarer un sinistre" in data["answer"]
    assert data["sources"] == ["Contexte factice n°1"]
    assert data["latency_ms"] >= 0


def test_ask_without_rag_has_no_sources(client):
    resp = client.post("/ask", json={"question": "Test", "use_rag": False})
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_ask_rejects_empty_question(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_ask_rejects_missing_question(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_without_override_returns_503():
    """Sans faux client injecté, /ask doit échouer proprement (503), pas planter."""
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        resp = c.post("/ask", json={"question": "Test"})
    assert resp.status_code == 503
