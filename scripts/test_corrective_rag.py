# scripts/test_corrective_rag.py
"""
Script de test manuel du graphe RAG correctif (LangGraph). Affiche
chaque étape pour voir le routage conditionnel en action -- notamment
si une reformulation ou une régénération a eu lieu.

Usage:
    python scripts/test_corrective_rag.py "Qu'est-ce qu'une franchise en assurance ?"
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.corrective_rag.graph import run_corrective_rag

DEFAULT_QUESTION = "Qu'est-ce qu'une franchise en assurance ?"


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question : {question}\n")

    result = run_corrective_rag(question)

    print(f"Requête finale utilisée pour le retrieval : {result['query']!r}")
    print(f"(reformulée : {result['query'] != question})")
    print(f"Documents pertinents jugés par le LLM : {result['pertinent']}")
    print(f"Réponse fondée sur le contexte (vérifiée) : {result['fondee']}")
    print(f"Nombre de tours (reformulation + régénération) : {result['tours']}")
    print("\n--- Documents récupérés (dernier tour) ---")
    for d in result["documents"]:
        print(f"  - {d[:100]}")
    print("\n--- Réponse finale ---")
    print(result["reponse"])


if __name__ == "__main__":
    main()
