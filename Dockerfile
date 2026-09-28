# Dockerfile -- API (api/main.py)
#
# Image CPU uniquement : le projet tourne sur CPU par contrainte
# matérielle (voir README, section Limites), pas la peine d'embarquer
# les paquets CUDA de PyTorch (plusieurs Go pour rien).

FROM python:3.11-slim

WORKDIR /app

# --- Couche dépendances -- rarement modifiée, la plus lente à construire.
# Copiée et installée AVANT le code : tant que requirements.txt ne change
# pas, Docker réutilise cette couche telle quelle sur les prochains builds.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- Couche code -- change à chaque commit, se reconstruit en quelques
# secondes grâce au cache de la couche précédente.
COPY api/ api/
COPY src/ src/

# Les poids du modèle et l'index RAG ne sont JAMAIS copiés dans l'image
# (ils sont gitignorés, potentiellement plusieurs Go) : ils arrivent par
# des volumes au lancement (voir docker-compose.yml).
# NB : PEFT_MODEL_DIR pointe vers .../models/... car le dossier local
# s'appelle "models/models/assurance-lora-v3" (double "models", hérité
# des sessions précédentes) -- pas une faute de frappe ici.
ENV HF_HOME=/data/hf-cache \
    PEFT_MODEL_DIR=/data/models/models/assurance-lora-v3 \
    RAG_INDEX_DIR=/data/faiss_index \
    PYTHONUNBUFFERED=1

# Utilisateur non-root : une faille exploitée dans le conteneur n'a pas
# les droits administrateur du conteneur.
RUN useradd --create-home appuser \
    && mkdir -p /data/hf-cache /data/models /data/faiss_index \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
