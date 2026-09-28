# scripts/publish_to_hub.py
"""
Publie l'adaptateur LoRA et l'index RAG sur le Hugging Face Hub, pour
qu'un environnement sans accès au disque local (Cloud Run, chapitre 3)
puisse les récupérer au démarrage.

- L'adaptateur : dépôt de type "model" -- PeftModel.from_pretrained()
  saura le charger directement via son repo_id, aucun code supplémentaire
  n'est nécessaire côté src/llm_client.py.
- L'index RAG : dépôt de type "dataset" -- src/rag_pipeline.py sait déjà
  le retélécharger via ensure_index_available() si RAG_INDEX_HUB_REPO
  est défini et que les fichiers ne sont pas déjà présents localement.

Prérequis : être connecté (`huggingface-cli login`, voir le README).

Usage:
    python scripts/publish_to_hub.py --username <ton-nom-huggingface>
"""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from huggingface_hub import HfApi, whoami

ADAPTER_DIR = os.path.join(ROOT_DIR, "models", "models", "assurance-lora-v3")
INDEX_DIR = os.path.join(ROOT_DIR, "data", "faiss_index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default=None, help="Nom d'utilisateur Hugging Face (déduit du login si omis)")
    parser.add_argument("--private", action="store_true", help="Créer les dépôts en privé (par défaut : public)")
    args = parser.parse_args()

    api = HfApi()
    username = args.username or whoami()["name"]

    adapter_repo = f"{username}/assurance-lora-v3"
    index_repo = f"{username}/assurance-rag-index"

    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(f"{ADAPTER_DIR} introuvable -- lance scripts/train_lora.py d'abord.")
    if not os.path.isdir(INDEX_DIR):
        raise FileNotFoundError(f"{INDEX_DIR} introuvable -- lance scripts/build_rag_index.py d'abord.")

    print(f"--- Adaptateur LoRA -> {adapter_repo} ({'privé' if args.private else 'public'}) ---")
    api.create_repo(adapter_repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=adapter_repo, repo_type="model", folder_path=ADAPTER_DIR)
    print(f"OK : https://huggingface.co/{adapter_repo}")

    print(f"--- Index RAG -> {index_repo} ({'privé' if args.private else 'public'}) ---")
    api.create_repo(index_repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(
        repo_id=index_repo, repo_type="dataset",
        path_or_fileobj=os.path.join(INDEX_DIR, "index.faiss"), path_in_repo="index.faiss",
    )
    api.upload_file(
        repo_id=index_repo, repo_type="dataset",
        path_or_fileobj=os.path.join(INDEX_DIR, "corpus.json"), path_in_repo="corpus.json",
    )
    print(f"OK : https://huggingface.co/datasets/{index_repo}")

    print()
    print("Variables d'environnement à utiliser pour le déploiement Cloud Run :")
    print(f"  PEFT_MODEL_DIR={adapter_repo}")
    print(f"  RAG_INDEX_HUB_REPO={index_repo}")


if __name__ == "__main__":
    main()
