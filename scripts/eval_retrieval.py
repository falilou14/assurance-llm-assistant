# scripts/eval_retrieval.py
"""
Évaluation recall@k du retrieval RAG, automatisée sur tout le dataset
Q/R (43 paires) -- remplace le test manuel sur 4 questions (voir la
fiche d'étude, section évaluation) qui datait de l'ancien corpus à 15
chunks, avant la reconstruction du dataset.

Méthode -- "self-retrieval" : chaque paire (instruction, input) du
dataset donne une question ET son contexte source exact (le "ground
truth"). On interroge FAISS avec la question, et on vérifie si le
contexte source apparaît dans les top-k résultats. Standard quand on
n'a pas d'annotation séparée -- le dataset d'entraînement sert aussi
de jeu d'évaluation du retrieval (deux usages différents des mêmes
données : entraîner le modèle vs vérifier que le bon passage est
retrouvable).

Usage:
    python scripts/eval_retrieval.py
"""

import json
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.rag_pipeline import RAGPipeline

DATASET_PATH = os.path.join(ROOT_DIR, "data", "processed", "dataset.jsonl")
INDEX_DIR = os.path.join(ROOT_DIR, "data", "faiss_index")
K_VALUES = [1, 2, 3, 5]


def load_eval_pairs():
    """
    Ne garde que les paires avec un contexte source non vide. Le dataset
    contient volontairement des exemples sans grounding (questions
    hors-sujet ou plaintes génériques, "input" vide -- entraînent le
    modèle à refuser/gérer sans RAG, pas à retrouver un passage). Le
    recall@k n'a pas de sens pour elles : il n'y a rien à retrouver.
    """
    pairs = []
    skipped = 0
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            context = row["input"].strip()
            if not context:
                skipped += 1
                continue
            pairs.append((row["instruction"], context))
    if skipped:
        print(f"({skipped} paire(s) sans contexte source exclue(s) -- hors-sujet/refus, pas du retrieval)\n")
    return pairs


def main():
    pairs = load_eval_pairs()
    max_k = max(K_VALUES)
    rag = RAGPipeline(INDEX_DIR)

    hits_at_k = {k: 0 for k in K_VALUES}
    misses = []

    for question, ground_truth in pairs:
        results = rag.query(question, top_k=max_k)
        retrieved_texts = [r["text"].strip() for r in results]

        # Rang (1-indexé) du 1er passage qui correspond exactement au
        # contexte source -- None si absent des max_k résultats.
        rank = next(
            (i + 1 for i, text in enumerate(retrieved_texts) if text == ground_truth),
            None,
        )
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
        if rank is None or rank > 1:
            misses.append((question, rank))

    n = len(pairs)
    print(f"Dataset : {n} paires question/contexte ({DATASET_PATH})")
    print(f"Corpus FAISS : {len(rag.corpus)} chunks ({INDEX_DIR})\n")
    print("k\trecall@k\thits/total")
    for k in K_VALUES:
        h = hits_at_k[k]
        print(f"{k}\t{h/n:.1%}\t\t{h}/{n}")

    if misses:
        print(f"\n{len(misses)} question(s) où le contexte source n'est PAS le 1er résultat :")
        for q, rank in misses:
            rank_str = f"rang {rank}" if rank else "absent du top-5"
            print(f"  - [{rank_str}] {q}")


if __name__ == "__main__":
    main()
