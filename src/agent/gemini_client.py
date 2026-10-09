# src/agent/gemini_client.py
"""
Fine couche autour du SDK google-genai -- ne contient QUE l'appel
réseau (démarrer/poursuivre une "interaction"). La boucle ReAct
(décider d'exécuter un outil, valider ses arguments, boucler) est dans
src/agent/loop.py, volontairement séparée : ce fichier ne connaît rien
du domaine assurance ni des outils eux-mêmes.

Utilise l'Interactions API (client.interactions.create), l'interface
actuelle du SDK google-genai pour le function calling -- pas
l'ancienne API generate_content/FunctionDeclaration, dépréciée dans la
documentation au moment où ce code a été écrit (29 sept 2026).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")

# Par défaut, le SDK retente jusqu'à 5 fois sur une erreur 5xx/429, avec un
# backoff qui peut monter à 60s entre tentatives -- sans limite de temps par
# requête individuelle. Sur le palier gratuit, un 503 ("Service Unavailable",
# surcharge côté Google) peut donc bloquer une requête plusieurs minutes.
# On borne explicitement les deux : 20s max par tentative HTTP, 2 tentatives
# au total -- un appel de l'agent échoue vite et proprement plutôt que de
# faire attendre l'utilisateur indéfiniment (même logique que MAX_TURNS dans
# loop.py, mais côté réseau plutôt que côté boucle).
_HTTP_OPTIONS = types.HttpOptions(
    timeout=20_000,  # millisecondes
    retry_options=types.HttpRetryOptions(attempts=2),
)


class GeminiAgentClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY manquant -- copie .env.example vers .env et "
                "renseigne ta clé (gratuite sur https://aistudio.google.com/apikey)."
            )
        self.client = genai.Client(api_key=api_key, http_options=_HTTP_OPTIONS)
        self.model = model

    def start(self, user_input: str, tools: list[dict]):
        """Démarre une nouvelle interaction (premier tour)."""
        return self.client.interactions.create(
            model=self.model,
            input=user_input,
            tools=tools,
        )

    def continue_with_results(
        self, previous_interaction_id: str, function_results: list[dict], tools: list[dict]
    ):
        """Renvoie le(s) résultat(s) d'outil au modèle pour poursuivre l'interaction."""
        return self.client.interactions.create(
            model=self.model,
            input=function_results,
            tools=tools,
            previous_interaction_id=previous_interaction_id,
        )
