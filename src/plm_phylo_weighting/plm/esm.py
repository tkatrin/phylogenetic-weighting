from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def clean_sequence(sequence: str) -> str:
    return sequence.replace("-", "").replace(".", "").replace("*", "").upper()


def load_plm(model_name: str, device: str) -> tuple[AutoTokenizer, AutoModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def get_sequence_embedding(
    sequence: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    max_length: int = 1024,
) -> np.ndarray:
    sequence = clean_sequence(sequence)

    if not sequence:
        return np.zeros(model.config.hidden_size, dtype=float)

    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}
    outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]
    attention_mask = inputs["attention_mask"][0].bool()
    token_ids = inputs["input_ids"][0]

    special_tokens = {
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    }

    valid_mask = attention_mask.clone()
    for token_id in special_tokens:
        if token_id is not None:
            valid_mask &= token_ids != token_id

    if valid_mask.sum().item() == 0:
        embedding = hidden[attention_mask].mean(dim=0)
    else:
        embedding = hidden[valid_mask].mean(dim=0)

    return embedding.detach().cpu().numpy()
