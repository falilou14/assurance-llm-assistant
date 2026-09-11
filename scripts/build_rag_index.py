"""
Construit l'index FAISS pour le RAG à partir des contextes factuels déjà
présents dans data/processed/dataset.jsonl (champ "input" -- le contexte
source de chaque paire question/réponse, cf. scripts/build_qa_dataset.py).
Garantit une base cohérente entre RAG et training : ce sont les mêmes
phrases sources qui servaient de contexte à l'entraînement.

Usage:
    python scripts/build_rag_index.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.utils import read_jsonl
from src.rag_pipeline import build_faiss_index

DATASET_PATH = os.path.join(ROOT_DIR, "data", "processed", "dataset.jsonl")
INDEX_DIR = os.path.join(ROOT_DIR, "data", "faiss_index")


def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} introuvable. Lance d'abord notebooks/01_preprocessing.ipynb."
        )

    # On indexe les contextes factuels (champ "input"), pas les réponses
    # rédigées (champ "output") -- c'est ce type de texte que le RAG doit
    # retrouver et injecter dans le prompt. Dédoublonné, et les exemples
    # sans contexte (plaintes/refus) sont ignorés.
    seen = set()
    corpus_texts = []
    for example in read_jsonl(DATASET_PATH):
        ctx = example.get("input", "").strip()
        if ctx and ctx not in seen:
            seen.add(ctx)
            corpus_texts.append(ctx)
    print(f"{len(corpus_texts)} contextes uniques chargés depuis {DATASET_PATH}")

    build_faiss_index(corpus_texts, INDEX_DIR)
    print(f"Index FAISS construit dans {INDEX_DIR}")


if __name__ == "__main__":
    main()
