# src/corrective_rag/graph.py
"""
Le graphe RAG correctif (voir Dossier-Etude-LLM/langgraph-rag-correctif.html
pour le plan complet) :

    retrieve -> grade_documents -> [pertinent ? generate : rewrite_query -> retrieve]
    generate -> verify_answer -> [fondée ? FIN : generate]

Un seul compteur "tours" partagé entre les deux boucles (reformulation
ET régénération) -- même garde-fou que MAX_TURNS pour l'agent (chapitre
4) : sans plafond, une question hors corpus ferait boucler le graphe
indéfiniment.
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.corrective_rag.gemini_judge import safe_ask
from src.rag_pipeline import RAGPipeline

MAX_TOURS = 3
TOP_K = 3

# generate() utilise notre propre modèle fine-tuné (TinyLlama+LoRA), pas
# Gemini : c'est exactement la tâche pour laquelle il a été entraîné,
# contrairement aux 3 autres nœuds (grade_documents, rewrite_query,
# verify_answer) qui demandent un raisonnement/jugement structuré --
# même limite de capacité que celle qui justifie Gemini pour l'agent
# (chapitre 4). Aucun téléchargement supplémentaire : les poids sont
# déjà en local depuis le chapitre 1.
PEFT_MODEL_DIR = os.environ.get("PEFT_MODEL_DIR", "models/models/assurance-lora-v3")
BASE_MODEL = os.environ.get(
    "BASE_MODEL", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)


class CorrectiveRAGState(TypedDict):
    question: str  # ne change jamais -- la vraie question de l'utilisateur
    query: str  # ce qui est effectivement cherché -- peut être reformulé
    documents: list[str]
    pertinent: bool
    reponse: str
    fondee: bool
    tours: int  # compteur partagé des deux boucles, plafonné à MAX_TOURS


_rag = None
_local_model = None  # (model, tokenizer, pipe) -- chargé une seule fois, voir generate()


def _get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        index_dir = os.environ.get("RAG_INDEX_DIR", "data/faiss_index")
        _rag = RAGPipeline(index_dir)
    return _rag


def _get_local_model():
    global _local_model
    if _local_model is None:
        from src.inference import load_model_for_inference  # import lourd -- seulement ici

        _local_model = load_model_for_inference(PEFT_MODEL_DIR, BASE_MODEL)
    return _local_model


# --- Nœuds -------------------------------------------------------------


def retrieve(state: CorrectiveRAGState) -> dict:
    print(f"  [noeud] retrieve (tours={state['tours']}) query={state['query']!r}", flush=True)
    hits = _get_rag().query(state["query"], top_k=TOP_K)
    return {"documents": [hit["text"] for hit in hits]}


def grade_documents(state: CorrectiveRAGState) -> dict:
    print(f"  [noeud] grade_documents (tours={state['tours']})", flush=True)
    contexte = "\n".join(f"- {d}" for d in state["documents"]) or "(aucun passage trouvé)"
    prompt = (
        f"Question : {state['question']}\n\nPassages récupérés :\n{contexte}\n\n"
        "Ces passages permettent-ils de répondre correctement à la question ? "
        "Réponds UNIQUEMENT par le mot OUI ou le mot NON, rien d'autre."
    )
    # Échec de l'appel -> "NON" par défaut : ça déclenche naturellement la
    # boucle de reformulation existante, plutôt qu'un plantage. Le plafond
    # de tours garantit qu'on ne boucle jamais indéfiniment pour autant.
    verdict = safe_ask(prompt, default="NON").strip().upper()
    return {"pertinent": verdict.startswith("OUI")}


def rewrite_query(state: CorrectiveRAGState) -> dict:
    print(f"  [noeud] rewrite_query (tours={state['tours']})", flush=True)
    prompt = (
        "Reformule cette question d'assurance avec un vocabulaire plus "
        "proche d'une documentation officielle (termes techniques précis "
        f"plutôt que langage courant). Question originale : {state['question']}\n"
        "Ne renvoie QUE la question reformulée, rien d'autre."
    )
    # Échec -> on garde la question d'origine plutôt que planter.
    nouvelle_query = safe_ask(prompt, default=state["question"]).strip() or state["question"]
    return {"query": nouvelle_query, "tours": state["tours"] + 1}


def generate(state: CorrectiveRAGState) -> dict:
    print(f"  [noeud] generate (tours={state['tours']}, modèle local)", flush=True)
    from src.inference import build_prompt, run_generation  # import lourd -- seulement ici

    contexte = "\n".join(state["documents"])
    _, _, pipe = _get_local_model()
    prompt = build_prompt(state["question"], contexte)
    try:
        reponse = run_generation(pipe, prompt, max_new_tokens=200)
    except Exception as exc:  # même filet que pour Gemini -- jamais planter le graphe
        print(f"[generate] échec du modèle local, on continue en dégradé : {exc}")
        reponse = "Désolé, une erreur technique empêche de répondre pour le moment."
    return {"reponse": reponse, "tours": state["tours"] + 1}


def verify_answer(state: CorrectiveRAGState) -> dict:
    print(f"  [noeud] verify_answer (tours={state['tours']})", flush=True)
    contexte = "\n".join(f"- {d}" for d in state["documents"]) or "(aucun passage trouvé)"
    prompt = (
        f"Contexte :\n{contexte}\n\nRéponse donnée : {state['reponse']}\n\n"
        "Cette réponse est-elle fondée UNIQUEMENT sur le contexte ci-dessus, "
        "sans affirmation inventée ? Réponds UNIQUEMENT par OUI ou NON."
    )
    # Échec -> "NON" par défaut : déclenche une régénération (ou l'arrêt
    # propre si le plafond est déjà atteint), jamais un plantage.
    verdict = safe_ask(prompt, default="NON").strip().upper()
    return {"fondee": verdict.startswith("OUI")}


# --- Routage conditionnel ------------------------------------------------


def route_after_grade(state: CorrectiveRAGState) -> str:
    if state["pertinent"]:
        return "generate"
    if state["tours"] >= MAX_TOURS:
        # Plafond atteint : on force une réponse (honnête, voir le prompt
        # de generate) plutôt que de reformuler indéfiniment.
        return "generate"
    return "rewrite_query"


def route_after_verify(state: CorrectiveRAGState) -> str:
    if state["fondee"] or state["tours"] >= MAX_TOURS:
        return END
    return "generate"


# --- Construction du graphe ----------------------------------------------


def build_graph():
    graph = StateGraph(CorrectiveRAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)
    graph.add_node("verify_answer", verify_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "verify_answer")
    graph.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {END: END, "generate": "generate"},
    )

    return graph.compile()


def run_corrective_rag(question: str) -> CorrectiveRAGState:
    app = build_graph()
    initial_state: CorrectiveRAGState = {
        "question": question,
        "query": question,
        "documents": [],
        "pertinent": False,
        "reponse": "",
        "fondee": False,
        "tours": 0,
    }
    return app.invoke(initial_state)
