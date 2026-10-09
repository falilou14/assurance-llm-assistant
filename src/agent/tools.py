# src/agent/tools.py
"""
Outils (tools) que l'agent peut appeler.

Volontairement séparé de la boucle de décision (src/agent/loop.py) :
le modèle ne fait que PROPOSER un nom d'outil + des arguments -- ce
fichier est le seul à décider si l'outil existe (liste blanche) et si
les arguments sont valides (Pydantic) avant de l'exécuter. On ne fait
jamais confiance à un argument renvoyé par le modèle sans validation,
exactement comme pour une entrée utilisateur brute côté API.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

_rag_pipeline = None


def _get_rag_pipeline():
    """Charge l'index RAG une seule fois (même logique que LocalLLMClient)."""
    global _rag_pipeline
    if _rag_pipeline is None:
        from src.rag_pipeline import RAGPipeline

        index_dir = os.environ.get("RAG_INDEX_DIR", "data/faiss_index")
        _rag_pipeline = RAGPipeline(index_dir)
    return _rag_pipeline


# --- 1. search_docs -----------------------------------------------------


class SearchDocsArgs(BaseModel):
    query: str = Field(
        ..., description="Question ou mots-clés à rechercher dans la documentation assurance."
    )
    top_k: int = Field(3, ge=1, le=5, description="Nombre de passages à renvoyer (1 à 5).")


def search_docs(query: str, top_k: int = 3) -> dict:
    hits = _get_rag_pipeline().query(query, top_k=top_k)
    return {"passages": [hit["text"] for hit in hits]}


# --- 2. calculer_franchise ------------------------------------------------

FRANCHISES = {"auto": 350.0, "habitation": 150.0, "sante": 50.0}


class CalculerFranchiseArgs(BaseModel):
    type_contrat: str = Field(
        ..., description="Type de contrat : 'auto', 'habitation' ou 'sante'."
    )
    montant_sinistre: float = Field(..., gt=0, description="Montant estimé du sinistre en euros.")


def calculer_franchise(type_contrat: str, montant_sinistre: float) -> dict:
    type_contrat = type_contrat.lower().strip()
    if type_contrat not in FRANCHISES:
        return {
            "erreur": f"Type de contrat inconnu : {type_contrat!r}. "
            f"Valeurs possibles : {list(FRANCHISES)}."
        }
    franchise = FRANCHISES[type_contrat]
    montant_rembourse = max(0.0, montant_sinistre - franchise)
    return {
        "franchise_appliquee": franchise,
        "montant_sinistre": montant_sinistre,
        "montant_rembourse": round(montant_rembourse, 2),
    }


# --- 3. verifier_delai_declaration ----------------------------------------

DELAIS_DECLARATION_JOURS = {
    "vol": 2,
    "degat_des_eaux": 5,
    "incendie": 5,
    "accident_auto": 5,
    "catastrophe_naturelle": 10,
}


class VerifierDelaiArgs(BaseModel):
    type_sinistre: str = Field(
        ...,
        description=(
            "Type de sinistre : 'vol', 'degat_des_eaux', 'incendie', "
            "'accident_auto' ou 'catastrophe_naturelle'."
        ),
    )
    jours_ecoules: int = Field(..., ge=0, description="Nombre de jours écoulés depuis le sinistre.")


def verifier_delai_declaration(type_sinistre: str, jours_ecoules: int) -> dict:
    type_sinistre = type_sinistre.lower().strip()
    if type_sinistre not in DELAIS_DECLARATION_JOURS:
        return {
            "erreur": f"Type de sinistre inconnu : {type_sinistre!r}. "
            f"Valeurs possibles : {list(DELAIS_DECLARATION_JOURS)}."
        }
    delai = DELAIS_DECLARATION_JOURS[type_sinistre]
    jours_restants = delai - jours_ecoules
    return {
        "delai_legal_jours": delai,
        "jours_ecoules": jours_ecoules,
        "encore_dans_les_delais": jours_restants >= 0,
        "jours_restants": max(0, jours_restants),
    }


# --- 4. preparer_declaration_sinistre --------------------------------------


class PreparerDeclarationArgs(BaseModel):
    nom_assure: str = Field(..., description="Nom complet de l'assuré.")
    numero_contrat: str = Field(..., description="Numéro du contrat d'assurance.")
    type_sinistre: str = Field(..., description="Type de sinistre déclaré.")
    description: str = Field(..., description="Description des faits en quelques phrases.")


def preparer_declaration_sinistre(
    nom_assure: str, numero_contrat: str, type_sinistre: str, description: str
) -> dict:
    # Simulé volontairement : ne soumet rien, ne modifie aucun état réel.
    # Un outil qui déclencherait une vraie action (envoi, appel à une API
    # tierce, écriture en base) devrait exiger une confirmation explicite
    # de l'utilisateur avant exécution -- pas seulement une décision du
    # modèle. Voir la fiche d'étude, chapitre 4, "outils à effet de bord".
    return {
        "statut": "brouillon_pret",
        "recapitulatif": {
            "assure": nom_assure,
            "contrat": numero_contrat,
            "type_sinistre": type_sinistre,
            "description": description,
        },
        "note": "Brouillon simulé -- aucune déclaration réelle n'a été envoyée.",
    }


# --- Registre + schéma JSON pour l'API Gemini -------------------------------

TOOL_SPECS: dict[str, dict] = {
    "search_docs": {
        "func": search_docs,
        "args_model": SearchDocsArgs,
        "description": (
            "Recherche des passages pertinents dans la documentation assurance "
            "(garanties, exclusions, procédures)."
        ),
    },
    "calculer_franchise": {
        "func": calculer_franchise,
        "args_model": CalculerFranchiseArgs,
        "description": (
            "Calcule la franchise applicable et le montant remboursé pour un sinistre donné."
        ),
    },
    "verifier_delai_declaration": {
        "func": verifier_delai_declaration,
        "args_model": VerifierDelaiArgs,
        "description": "Vérifie si un sinistre est encore dans les délais légaux de déclaration.",
    },
    "preparer_declaration_sinistre": {
        "func": preparer_declaration_sinistre,
        "args_model": PreparerDeclarationArgs,
        "description": (
            "Prépare un brouillon de déclaration de sinistre structuré "
            "(simulation, n'envoie rien réellement)."
        ),
    },
}


def _pydantic_to_gemini_schema(model: type[BaseModel]) -> dict:
    """
    Convertit un modèle Pydantic en schéma JSON minimal compatible avec
    les déclarations de fonctions de l'API Gemini (pas de $ref/$defs,
    pas de "title" -- on ne garde que type/description/enum).
    """
    schema = model.model_json_schema()
    properties = {}
    for name, prop in schema.get("properties", {}).items():
        entry = {"type": prop.get("type", "string")}
        if "description" in prop:
            entry["description"] = prop["description"]
        if "enum" in prop:
            entry["enum"] = prop["enum"]
        properties[name] = entry
    return {
        "type": "object",
        "properties": properties,
        "required": schema.get("required", []),
    }


def build_tool_declarations() -> list[dict]:
    """Construit la liste de déclarations d'outils envoyée à l'API Gemini."""
    return [
        {
            "type": "function",
            "name": name,
            "description": spec["description"],
            "parameters": _pydantic_to_gemini_schema(spec["args_model"]),
        }
        for name, spec in TOOL_SPECS.items()
    ]
