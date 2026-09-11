"""
Entraînement LoRA (QLoRA : base 4-bit + adaptateur LoRA) cohérent avec l'inférence.

Point clé de ce script (corrige les incohérences trouvées dans notebooks/02) :
- Le template de prompt utilisé ici (`format_conversation_prompt`, src/utils.py)
  est EXACTEMENT le même que celui utilisé à l'inférence (src/inference.py).
  Un LLM fine-tuné apprend une association forme-de-prompt -> forme-de-réponse
  très spécifique : changer le template entre train et inference casse cette
  association, même si le contenu sémantique est similaire.
- Les noms de champs (instruction/input/output) correspondent à ceux produits
  par src/dataset_builder.py -- pas de champs inventés ("response", etc.)

Usage:
    python scripts/train_lora.py
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

from src.utils import read_jsonl, format_conversation_prompt, set_seed

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATASET_PATH = os.path.join(ROOT_DIR, "data", "processed", "dataset.jsonl")
OUTPUT_DIR = os.path.join(ROOT_DIR, "models", "models", "assurance-lora-v3")
MAX_LEN = 256


def build_examples():
    examples = list(read_jsonl(DATASET_PATH))
    texts = []
    for ex in examples:
        # Même template qu'à l'inférence : le modèle apprend à continuer
        # après "### Réponse:\n" avec la réponse attendue.
        prompt = format_conversation_prompt(ex["instruction"], ex["input"], ex["output"])
        texts.append(prompt)
    return texts


def tokenize_fn(tokenizer):
    def _fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out
    return _fn


def main():
    set_seed(42)

    texts = build_examples()
    print(f"{len(texts)} exemples chargés depuis {DATASET_PATH}")
    print("--- Exemple de prompt d'entraînement ---")
    print(texts[0])
    print("-----------------------------------------")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = Dataset.from_dict({"text": texts})
    ds = ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=["text"])

    # Pas de quantization 4-bit ici : bitsandbytes sur CPU a un chemin de calcul
    # mal optimisé (lent, thrashing mémoire observé en pratique). Pour un aussi
    # petit modèle (1.1B) et un aussi petit dataset (15 exemples), charger en
    # bfloat16 direct est plus simple et plus rapide, quitte à consommer un peu
    # plus de RAM (~2.2 Go). La 4-bit reste pertinente pour l'INFERENCE seule
    # (voir src/lora_training.py) ou pour des modèles bien plus gros.
    print(f"Chargement du base model (bfloat16, CPU) : {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.enable_input_require_grads()  # nécessaire : le base model est gelé,
    # sans ça aucun gradient ne circule jusqu'aux matrices LoRA en amont.

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=6,  # 43 exemples (contre 15 en v2) -> moins d'epochs pour un temps total comparable
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Adaptateur sauvegardé dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
