# src/corrective_rag/gemini_judge.py
"""
Appels Gemini ponctuels, sans état ni outils -- utilisés par les nœuds
"juges" du graphe (grade_documents, verify_answer) et par generate/
rewrite_query. Volontairement séparé de src/agent/gemini_client.py :
celui-ci gère des conversations à outils (Interactions API + historique),
ici chaque appel est indépendant, une question entre, une réponse sort.

Même limite de temps/retries que pour l'agent (voir src/agent/gemini_client.py)
-- leçon du chapitre 4 : un appel externe sans plafond peut bloquer un
nœud du graphe indéfiniment si l'API tierce est lente ou en panne.
"""

from __future__ import annotations

import concurrent.futures
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Exécuteur partagé pour imposer un vrai plafond de temps côté CLIENT
# (voir safe_ask) -- le timeout configuré dans _HTTP_OPTIONS ci-dessous
# s'est révélé peu fiable en pratique sur cette API bêta (un appel est
# resté bloqué 115s+ sans jamais déclencher APITimeoutError, incident
# constaté le 5 oct. 2026). On ne peut pas annuler un appel réseau en
# cours sous Windows, mais on peut arrêter d'ATTENDRE après N secondes
# et traiter ça comme un échec -- le thread orphelin se termine plus
# tard dans le vide, sans bloquer le reste du programme.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

_HTTP_OPTIONS = types.HttpOptions(
    timeout=20_000,
    retry_options=types.HttpRetryOptions(attempts=2),
)

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY manquant -- copie .env.example vers .env et "
                "renseigne ta clé (gratuite sur https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=api_key, http_options=_HTTP_OPTIONS)
    return _client


def ask(prompt: str) -> str:
    """Un appel, une réponse texte -- pas de conversation, pas d'outils."""
    interaction = _get_client().interactions.create(model=MODEL, input=prompt)
    return (interaction.output_text or "").strip()


def safe_ask(prompt: str, default: str = "", timeout_s: float = 25.0) -> str:
    """
    Comme ask(), mais ne lève jamais et ne bloque jamais plus de
    timeout_s secondes -- un timeout/503 ponctuel de Gemini ne doit pas
    planter ni geler tout le graphe LangGraph. Deux incidents réels
    rencontrés en testant le RAG correctif (5 oct. 2026) : (1) un appel
    qui échoue dans rewrite_query faisait planter run_corrective_rag()
    entièrement -- corrigé par le try/except ci-dessous ; (2) le
    timeout CÔTÉ SDK (_HTTP_OPTIONS) ne s'est pas toujours déclenché --
    un appel est resté bloqué 115s+ sans jamais lever d'exception. D'où
    ce second filet, côté CLIENT : on n'attend jamais plus de
    timeout_s, qu'importe ce que fait le SDK en coulisses.
    """
    future = _executor.submit(ask, prompt)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        print(f"[gemini_judge] appel trop lent (>{timeout_s}s), on continue en dégradé")
        return default
    except Exception as exc:  # réseau, timeout SDK, 503 -- jamais une raison de planter
        print(f"[gemini_judge] appel échoué, on continue en dégradé : {exc}")
        return default
