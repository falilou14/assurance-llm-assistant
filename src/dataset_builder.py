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

    def save_examples(self, examples: List[Dict], output_path: str):
        """
        Sauvegarde une liste d'exemples déjà construits (instruction/input/output)
        en JSONL -- contrairement à save_dataset(), n'impose pas une instruction
        fixe : chaque exemple porte sa propre question/instruction, ce qui est
        indispensable pour qu'un fine-tuning apprenne à conditionner sa réponse
        sur la question posée plutôt que sur un style générique.
        """
        for ex in examples:
            missing = {"instruction", "input", "output"} - ex.keys()
            if missing:
                raise ValueError(f"Exemple incomplet, champs manquants: {missing} -> {ex}")

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
