# src/rag_pipeline.py
"""
RAG minimal : embeddings + FAISS index + retrieval.
- build_faiss_index(corpus_texts, index_dir)
- query_faiss(query, index_dir, top_k)
- helper to create corpus from data/raw or processed dataset
"""

import os
import json
from typing import List, Tuple
import numpy as np

# SentenceTransformers for embeddings
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_corpus_from_raw(raw_dir: str) -> List[str]:
    """
    Parcourt data/raw/ et renvoie une liste de textes (documents/chunks).
    """
    texts = []
    for filename in os.listdir(raw_dir):
        if filename.lower().endswith(".txt"):
            path = os.path.join(raw_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read().strip())
    return texts


def build_faiss_index(corpus_texts: List[str], index_dir: str, embedding_model_name: str = EMBED_MODEL):
    """
    Crée les embeddings pour corpus_texts, construit un index FAISS et sauvegarde :
    - index.faiss (binary)
    - corpus.json (texte + id)
    """
    os.makedirs(index_dir, exist_ok=True)
    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(corpus_texts, show_progress_bar=True, convert_to_numpy=True)

    # normalize embeddings if using cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (after L2 normalize -> cosine)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    # save corpus mapping
    corpus_path = os.path.join(index_dir, "corpus.json")
    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_texts, f, ensure_ascii=False, indent=2)

    return index_dir


def load_faiss_index(index_dir: str) -> Tuple[faiss.Index, List[str], SentenceTransformer]:
    """
    Charge index + corpus + embedder
    """
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    corpus = json.load(open(os.path.join(index_dir, "corpus.json"), "r", encoding="utf-8"))
    embedder = SentenceTransformer(EMBED_MODEL)
    return index, corpus, embedder


def query_faiss(query: str, index_dir: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
    """
    Recherche les top_k documents les plus pertinents pour la query.
    Retourne une liste de tuples (idx, score, text).
    """
    index, corpus, embedder = load_faiss_index(index_dir)
    q_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append((int(idx), float(score), corpus[idx]))
    return results



class RAGPipeline:
    def __init__(self, index_dir: str, model_name: str = EMBED_MODEL):
        self.index_dir = index_dir
        self.embedder = SentenceTransformer(model_name)
        self.index, self.corpus, _ = load_faiss_index(index_dir)

    def query(self, query: str, top_k: int = 3):
        q_vec = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                "id": int(idx),
                "score": float(score),
                "text": self.corpus[idx]
            })
        return results
