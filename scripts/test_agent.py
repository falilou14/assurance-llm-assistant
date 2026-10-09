# scripts/test_agent.py
"""
Script de test manuel pour l'agent (chapitre 4) -- pas encore branché
sur l'API FastAPI, ça viendra dans une étape suivante. Sert à vérifier
que la boucle ReAct + les 4 outils fonctionnent avant de les exposer en
HTTP.

Usage :
    python scripts/test_agent.py "Ma voiture a été volée il y a 3 jours, ..."
    python scripts/test_agent.py        # utilise une question par défaut
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.agent.loop import run_agent

DEFAULT_QUESTION = (
    "Ma voiture a été volée il y a 3 jours, le sinistre est estimé à 1200€. "
    "Quelle est la franchise et suis-je encore dans les délais pour déclarer ?"
)


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question : {question}\n")

    result = run_agent(question, verbose=True)

    print("\n--- Trace des outils utilisés ---")
    if not result["trace"]:
        print("  (aucun outil appelé)")
    for step in result["trace"]:
        print(f"  {step['outil']}({step['arguments']}) -> {step['resultat']}")

    print(f"\n--- Réponse finale ({result['nb_tours']} tour(s)) ---")
    print(result["reponse"])


if __name__ == "__main__":
    main()
