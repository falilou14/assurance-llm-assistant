# src/llm_client.py
"""
Backend "local" du LLM : charge le modèle une seule fois (TinyLlama +
adaptateur LoRA) et l'index RAG une seule fois, puis les réutilise pour
chaque requête.

C'est volontairement le SEUL fichier du projet qui importe des
dépendances lourdes (via src.inference -> torch/transformers, et
src.rag_pipeline -> sentence-transformers/faiss). L'API (api/) ne
l'importe jamais au niveau module -- seulement à l'intérieur de la
fonction de démarrage (lifespan, voir api/main.py) -- pour que les tests
automatiques de l'API n'aient jamais besoin de ces bibliothèques.
"""

import os

from src.inference import build_prompt, load_model_for_inference, run_generation
from src.rag_pipeline import RAGPipeline

PEFT_MODEL_DIR = os.environ.get("PEFT_MODEL_DIR", "models/models/assurance-lora-v3")
BASE_MODEL = os.environ.get(
    "BASE_MODEL", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)
RAG_INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "data/faiss_index")


class LocalLLMClient:
    """Implémente le contrat attendu par api/deps.py (méthode generate)."""

    def __init__(self):
        _, self.tokenizer, self.pipe = load_model_for_inference(PEFT_MODEL_DIR, BASE_MODEL)
        self.rag = RAGPipeline(RAG_INDEX_DIR)

    def generate(
        self, question: str, use_rag: bool, top_k: int, max_new_tokens: int
    ) -> tuple[str, list[str]]:
        sources: list[str] = []
        context = ""
        if use_rag:
            hits = self.rag.query(question, top_k=top_k)
            sources = [hit["text"] for hit in hits]
            context = "\n\n".join(sources)

        prompt = build_prompt(question, context)
        answer = run_generation(self.pipe, prompt, max_new_tokens=max_new_tokens)
        return answer, sources
