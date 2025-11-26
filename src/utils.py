# src/utils.py
"""
Petites fonctions utilitaires : prompt formatting, lecture JSONL, seed, etc.
"""

import json
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line.rstrip("\n"))


def save_jsonl(list_of_dicts, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in list_of_dicts:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def format_conversation_prompt(instruction: str, input_text: str, output_text: str):
    """
    Template simple : we build a single text containing instruction+input+response.
    This matches dataset_builder.build_instruction output.
    """
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text and input_text.strip():
        prompt += f"### Contexte:\n{input_text}\n"
    prompt += f"### Réponse:\n{output_text}\n"
    return prompt
