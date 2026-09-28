import json
import os

import requests
import streamlit as st

# ==========================================
# CONFIG STREAMLIT
# ==========================================
# Streamlit ne charge plus le modèle lui-même : il appelle l'API (api/main.py)
# en HTTP. Il faut donc lancer l'API séparément :
#   uvicorn api.main:app --port 8000
# avant (ou en même temps que) `streamlit run app/streamlit_app.py`.
# API_URL est surchargeable par variable d'environnement (utile en Docker,
# chapitre suivant, où l'API tourne dans un autre conteneur).

st.set_page_config(
    page_title="Assurance AI Assistant",
    page_icon="🛡️",
    layout="wide",
)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
RAG_INDEX_DIR = "data/faiss_index"  # encore utilisé ici juste pour lister les docs indexés (lecture locale)

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
            with open(corpus_path, encoding="utf-8") as f:
                n_docs = len(json.load(f))
        except Exception:
            n_docs = "?"

    # Interroge /ready plutôt que de supposer que l'API tourne -- l'appel
    # est rapide (pas de génération), on peut se permettre de le refaire
    # à chaque rendu de la page.
    try:
        r = requests.get(f"{API_URL}/ready", timeout=2)
        api_state = "🟢 prête" if r.ok and r.json().get("ready") else "🟠 en cours de chargement"
    except requests.exceptions.RequestException:
        api_state = "🔴 injoignable"

    st.markdown(f"""
    <div class="mini-metric"><span>API</span><span>{api_state}</span></div>
    <div class="mini-metric"><span>Modèle de base</span><span>TinyLlama-1.1B</span></div>
    <div class="mini-metric"><span>Adaptateur</span><span>lora-v3</span></div>
    <div class="mini-metric"><span>Chunks indexés</span><span>{n_docs}</span></div>
    """, unsafe_allow_html=True)
    if api_state == "🔴 injoignable":
        st.caption(f"Lance l'API : `uvicorn api.main:app --port 8000` (URL attendue : {API_URL})")

    if st.button("Voir les documents indexés"):
        if os.path.exists(corpus_path):
            with open(corpus_path, encoding="utf-8") as f:
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
# CHAT
# ==========================================
# Plus de chargement de modèle ici : l'API le fait une seule fois, au
# démarrage du serveur (voir api/main.py, lifespan). Streamlit ne fait
# plus qu'un appel HTTP par question.

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
        with st.spinner("Interrogation de l'API (génération CPU : jusqu'à 30-60 s)…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question,
                        "use_rag": use_rag,
                        "top_k": top_k,
                        "max_new_tokens": max_tokens,
                    },
                    timeout=180,  # génération CPU lente -- voir le dossier d'étude
                )
                if resp.status_code == 503:
                    answer = "⏳ Le modèle est encore en train de charger côté API. Réessaie dans quelques instants."
                elif not resp.ok:
                    answer = f"⚠️ Erreur API ({resp.status_code}) : {resp.text[:200]}"
                else:
                    data = resp.json()
                    answer = data["answer"]
                    if data.get("sources"):
                        previews = " · ".join(s[:60] + "…" for s in data["sources"][:3])
                        answer += f"\n\n---\n*Sources ({len(data['sources'])}, {data['latency_ms']:.0f} ms) : {previews}*"
            except requests.exceptions.RequestException as e:
                answer = f"🔴 API injoignable sur {API_URL} -- lance `uvicorn api.main:app --port 8000`.\n\n({e})"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

if not st.session_state.messages:
    st.markdown("""
    <div class="status-row">
      <span class="chip">💬 essaie : <b>"Comment déclarer un sinistre auto ?"</b></span>
      <span class="chip">💬 essaie : <b>"Puis-je résilier mon contrat ?"</b></span>
    </div>
    """, unsafe_allow_html=True)
