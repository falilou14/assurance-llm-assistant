# src/inference.py
"""
Chargement du modèle (base + LoRA) et génération.
Optionnellement combine RAG (retrieval) pour fournir du contexte.
"""

import os
from typing import Optional
import torch
from transformers import pipeline

from src.rag_pipeline import query_faiss
from src.utils import format_conversation_prompt
from src.lora_training import load_peft_model_bf16  # CPU-safe, sans 4-bit (voir notes)

DEVICE = "cpu"   # On force CPU, car load_peft_model_safe est CPU-only


# -----------------------------------------------------------
# 🔥 Chargement du modèle (LoRA + Base)
# -----------------------------------------------------------
def load_model_for_inference(peft_dir: str, base_model: Optional[str] = None):
    """
    Retourne (model, tokenizer, generator_pipe)
    - peft_dir : dossier des poids LoRA
    - base_model : nom du modèle de base (ex: mistralai/Mistral-7B-Instruct-v0.2)
    """

    # bf16 direct (pas de 4-bit) -> voir load_peft_model_bf16 pour le pourquoi
    model, tokenizer = load_peft_model_bf16(
        peft_dir=peft_dir,
        base_model_name=base_model,
    )

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        truncation=True
)


    return model, tokenizer, gen


# -----------------------------------------------------------
# 🔥 Génération de texte
# -----------------------------------------------------------
def generate_answer(
    question: str,
    model_pipeline,
    tokenizer,
    rag_index_dir: Optional[str] = None,
    top_k: int = 3,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
):
    """
    Génère une réponse.
    Si rag_index_dir fourni → ajout de contexte RAG.
    """

    # Même template qu'à l'entraînement (format_conversation_prompt) : un LLM
    # fine-tuné associe une forme de prompt précise à une forme de réponse --
    # changer le template entre train et inference casse cette association
    # (voir le bug de génération dégénérée diagnostiqué en session 1).
    if rag_index_dir:
        hits = query_faiss(question, rag_index_dir, top_k=top_k)
        context = "\n\n".join(hit[2] for hit in hits)
    else:
        context = ""

    prompt = format_conversation_prompt(
        instruction=question,
        input_text=context,
        output_text="",
    )
    # On génère à partir d'où le training s'arrêtait juste avant la réponse :
    # format_conversation_prompt termine toujours par "### Réponse:\n".

    outputs = model_pipeline(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    full_text = outputs[0]["generated_text"]
    # Le pipeline renvoie prompt + complétion concaténés ; on ne garde que
    # ce qui a été généré après "### Réponse:\n" pour l'affichage.
    return full_text[len(prompt):].strip()


# -----------------------------------------------------------
# 🔥 Helper pour Streamlit
# -----------------------------------------------------------
def answer_with_optional_rag(
    question: str,
    peft_dir: str,
    base_model: Optional[str],
    rag_dir: Optional[str] = None
):
    model, tokenizer, gen = load_model_for_inference(peft_dir, base_model)
    answer = generate_answer(question, gen, tokenizer, rag_index_dir=rag_dir)
    return answer
