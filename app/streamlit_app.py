import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
from src.inference import answer_with_optional_rag
import os

# ==========================================
# CONFIG STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Assurance AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Assistant IA – Assurance & Sinistres")

st.markdown("""
Bienvenue sur ton assistant IA spécialisé dans **l’assurance**.  
Il utilise un modèle **LoRA fine-tuné** + un système **RAG (FAISS)** pour répondre précisément.
""")

# ==========================================
# CHEMINS MODELE ET RAG
# ==========================================

PEFT_MODEL_DIR = "models/models/assurance-lora"   # adapter LoRA entraîné
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct" # ou ton modèle local
RAG_INDEX_DIR = "data/faiss_index"    # si tu veux activer RAG

# ==========================================
# INTERFACE
# ==========================================

with st.sidebar:
    st.header("⚙️ Options du modèle")

    use_rag = st.checkbox("Activer RAG (conseillé)", value=True)

    temperature = st.slider("Température", 0.0, 1.5, 0.3, 0.1)
    max_tokens = st.slider("Max tokens", 50, 500, 250, 10)

    st.divider()
    st.header("📁 Documents RAG")

    if st.button("Afficher les documents indexés"):
        if os.path.exists(RAG_INDEX_DIR):
            corpus_path = os.path.join(RAG_INDEX_DIR, "corpus.json")
            if os.path.exists(corpus_path):
                import json
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                st.write(corpus)
            else:
                st.warning("Aucun corpus trouvé dans l'index FAISS.")
        else:
            st.warning("Aucun index FAISS trouvé.")

# ==========================================
# CHAT UI
# ==========================================

user_input = st.text_area("💬 Pose ta question :", height=120)

if st.button("Envoyer"):
    if user_input.strip() == "":
        st.warning("Entre une question avant d'envoyer.")
    else:
        with st.spinner("Analyse en cours..."):
            # Appel correct de la fonction
            response = answer_with_optional_rag(
                question=user_input,
                peft_dir=PEFT_MODEL_DIR,
                base_model=BASE_MODEL,
                rag_dir=RAG_INDEX_DIR if use_rag else None
            )

        st.success("Réponse générée :")
        st.write(response)
