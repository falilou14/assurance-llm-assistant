# src/lora_training.py
"""
Version SAFE : 
- charge les modèles LoRA en 4-bit sur CPU UNIQUEMENT
- aucune utilisation de device_map="auto"
- aucune tentative d'offload sur disque
→ empêche totalement l'erreur : 
"You are trying to offload the whole model to the disk"
"""

import os
import logging
import torch
from typing import Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftConfig, PeftModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------
# 🔥 Chargement LoRA — CPU SAFE 4-bit
# ---------------------------------------------------
def load_peft_model_safe(peft_dir: str, base_model_name: Optional[str] = None):
    """
    Charge un modèle LoRA en 4-bit EN TOUTE SÉCURITÉ (CPU only).

    - Aucun offload
    - Aucun device_map="auto"
    - Aucun dispatch_model()
    """

    # Charger la configuration LoRA
    try:
        pconfig = PeftConfig.from_pretrained(peft_dir)
        if base_model_name is None:
            base_model_name = pconfig.base_model_name_or_path
    except:
        pconfig = None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name if base_model_name else peft_dir
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,      # Phi-3 préfère sans double quant
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # recommandé pour Phi3
    )


    logger.info(f"Chargement du base model 4-bit CPU : {base_model_name}")

    # IMPORTANT : PAS DE device_map
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Appliquer LoRA si existant
    if pconfig:
        logger.info(f"Application LoRA depuis {peft_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            peft_dir,
        )
    else:
        model = base_model

    return model, tokenizer
