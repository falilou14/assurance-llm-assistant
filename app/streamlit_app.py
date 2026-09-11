import sys
import os
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
from src.inference import load_model_for_inference, generate_answer

# ==========================================
# CONFIG STREAMLIT
# ==========================================

st.set_page_config(
    page_title="Assurance AI Assistant",
    page_icon="🛡️",
    layout="wide",
)

PEFT_MODEL_DIR = "models/models/assurance-lora-v3"   # v3 : dataset Q/R varié (43 ex.), plus l'ancien style figé
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
RAG_INDEX_DIR = "data/faiss_index"

# ==========================================
# STYLE — identité visuelle "dossier" (cohérente avec le carnet de notes)
# ==========================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
  --paper:#F6F6F0; --paper-raised:#EEEFE8; --ink:#1E2422; --ink-soft:#5B635C;
  --gold:#8A5E1F; --gold-soft:#B8862B; --teal:#2C5C55; --teal-soft:#3A6B63; --rule:#C9C4B4;
}

html, body, [class*="css"]{ font-family:'IBM Plex Sans', sans-serif; }

/* bannière d'en-tête */
.app-header{
  display:flex; align-items:center; gap:16px;
  padding:18px 24px; margin-bottom:8px;
  background:var(--paper-raised);
  border:1px solid var(--rule);
  border-radius:6px;
}
.app-header .badge{
  font-family:'IBM Plex Mono', monospace; font-size:26px;
  width:48px; height:48px; display:flex; align-items:center; justify-content:center;
  background:var(--teal); color:#fff; border-radius:6px; flex:none;
}
.app-header h1{
  font-family:'Fraunces', serif; font-weight:600; font-size:24px; margin:0; color:var(--ink);
}
.app-header p{ margin:2px 0 0; color:var(--ink-soft); font-size:13.5px; }

/* status chips sous le header */
.status-row{ display:flex; gap:10px; margin:10px 0 22px; flex-wrap:wrap; }
.chip{
  font-family:'IBM Plex Mono', monospace; font-size:11.5px; letter-spacing:.03em;
  padding:5px 11px; border-radius:20px; border:1px solid var(--rule);
  background:var(--paper-raised); color:var(--ink-soft);
}
.chip b{ color:var(--teal-soft); }

/* bulles de chat */
[data-testid="stChatMessage"]{
  border-radius:10px; border:1px solid var(--rule); background:var(--paper-raised);
}

/* sidebar */
section[data-testid="stSidebar"]{
  background:var(--paper-raised); border-right:1px solid var(--rule);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
  font-family:'IBM Plex Mono', monospace; font-size:12px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--gold-soft);
}

/* petites métriques sidebar */
.mini-metric{
  display:flex; justify-content:space-between; font-size:13px;
  padding:6px 0; border-bottom:1px solid var(--rule);
}
.mini-metric span:last-child{ font-family:'IBM Plex Mono', monospace; color:var(--teal-soft); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <div class="badge">🛡️</div>
  <div>
    <h1>Assistant Assurance & Sinistres</h1>
    <p>Modèle LoRA fine-tuné (QLoRA) + RAG (FAISS) sur tes documents d'assurance.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.header("Options")
    use_rag = st.checkbox("Activer le RAG", value=True, help="Injecte les passages les plus pertinents de tes documents dans le prompt.")
    top_k = st.slider("Documents récupérés (top_k)", 1, 5, 3)
    max_tokens = st.slider("Longueur max de la réponse", 50, 400, 200, 10)

    st.divider()
    st.header("État du système")

    n_docs = "—"
    corpus_path = os.path.join(RAG_INDEX_DIR, "corpus.json")
    if os.path.exists(corpus_path):
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                n_docs = len(json.load(f))
        except Exception:
            n_docs = "?"

    st.markdown(f"""
    <div class="mini-metric"><span>Modèle de base</span><span>TinyLlama-1.1B</span></div>
    <div class="mini-metric"><span>Adaptateur</span><span>lora-v3</span></div>
    <div class="mini-metric"><span>Chunks indexés</span><span>{n_docs}</span></div>
    <div class="mini-metric"><span>Device</span><span>CPU</span></div>
    """, unsafe_allow_html=True)

    if st.button("Voir les documents indexés"):
        if os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
            for i, chunk in enumerate(corpus):
                st.caption(f"#{i} — {chunk[:90]}…")
        else:
            st.warning("Aucun index FAISS trouvé — lance scripts/build_rag_index.py")

    st.divider()
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# CHARGEMENT DU MODÈLE — mis en cache (une seule fois par session serveur)
# ==========================================

@st.cache_resource(show_spinner=False)
def get_model():
    _, tokenizer, gen = load_model_for_inference(PEFT_MODEL_DIR, BASE_MODEL)
    return tokenizer, gen

# ==========================================
# CHAT
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🛡️"):
        st.markdown(msg["content"])

question = st.chat_input("Pose ta question sur ton assurance…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Chargement du modèle (jusqu'à ~1 min la 1ère fois) puis génération…"):
            tokenizer, gen = get_model()
            answer = generate_answer(
                question,
                gen,
                tokenizer,
                rag_index_dir=RAG_INDEX_DIR if use_rag else None,
                top_k=top_k,
                max_new_tokens=max_tokens,
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.messages:
    st.markdown("""
    <div class="status-row">
      <span class="chip">💬 essaie : <b>"Comment déclarer un sinistre auto ?"</b></span>
      <span class="chip">💬 essaie : <b>"Puis-je résilier mon contrat ?"</b></span>
    </div>
    """, unsafe_allow_html=True)
