from __future__ import annotations

import gc

import numpy as np
import torch

from plm_phylo_weighting.plm.cache import embeddings_exist, save_family_embeddings
from plm_phylo_weighting.plm.esm import get_sequence_embedding, load_plm
from plm_phylo_weighting.utils.config import get_device


def compute_embeddings_for_family(
    fam_id: str,
    fam: dict,
    tokenizer,
    model,
    model_key: str,
    work_dir: str,
    device: str,
    max_length: int,
    force: bool = False,
) -> None:
    if embeddings_exist(work_dir, fam_id, model_key) and not force:
        return

    alignment = fam["ASQ"]
    sequence_names = list(alignment.keys())

    embeddings = []

    for i, name in enumerate(sequence_names):
        if i % 100 == 0:
            print(model_key, fam_id, i, "/", len(sequence_names))

        embeddings.append(
            get_sequence_embedding(
                alignment[name],
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_length,
            )
        )

    save_family_embeddings(
        work_dir=work_dir,
        fam_id=fam_id,
        sequence_names=sequence_names,
        embeddings=np.vstack(embeddings),
        model_key=model_key,
    )


def compute_all_embeddings(config: dict, families: dict, valid_family_ids: list[str]) -> None:
    device = get_device(config)

    for model_key in config["model_keys_to_run"]:
        model_name = config["models"][model_key]
        print("loading", model_key, model_name)

        tokenizer, model = load_plm(model_name, device=device)

        for fam_id in valid_family_ids:
            compute_embeddings_for_family(
                fam_id=fam_id,
                fam=families[fam_id],
                tokenizer=tokenizer,
                model=model,
                model_key=model_key,
                work_dir=config["work_dir"],
                device=device,
                max_length=config["max_length"],
                force=False,
            )

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        del model
        del tokenizer

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
