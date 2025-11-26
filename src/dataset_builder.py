# src/dataset_builder.py

import json
from typing import List, Dict

class DatasetBuilder:
    """
    Transforme des chunks de texte en dataset JSONL au format instruction-based.
    """

    def __init__(self):
        pass

    def build_instruction(self, chunk: str) -> Dict:
        """
        Convertit un texte brut en un exemple "instruction / response".

        Exemple :
        - instruction: "Explique ce texte en termes simples"
        - response: <chunk>

        On peut changer l'instruction selon le domaine.
        """
        return {
            "instruction": "Explique en termes simples ce texte issu d’un contrat d’assurance.",
            "input": "",
            "output": chunk
        }

    def save_dataset(self, chunks: List[str], output_path: str):
        """
        Sauvegarde la liste des chunks en JSONL
        """
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                example = self.build_instruction(chunk)
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
