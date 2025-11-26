# src/preprocessing.py

import re
from typing import List

class TextPreprocessor:
    """
    Nettoyage et transformation du texte provenant du corpus brut.
    Ce module est utilisé pour normaliser le texte avant de créer le dataset.
    """

    def __init__(self, min_length: int = 20, max_length: int = 500):
        self.min_length = min_length
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte :
        - enlève les caractères inutiles
        - supprime les espaces multiples
        - normalise la ponctuation
        """
        text = text.strip()

        # Normalisation simple
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s,.;:!?()/%-]", "", text)

        return text

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Découpe un long texte en plusieurs morceaux (chunks)
        adaptés pour le fine-tuning ou le RAG.
        """
        text = self.clean_text(text)

        sentences = re.split(r"[.!?]", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) < self.max_length:
                current_chunk += sentence + ". "
            else:
                if len(current_chunk) >= self.min_length:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if len(current_chunk) >= self.min_length:
            chunks.append(current_chunk.strip())

        return chunks
