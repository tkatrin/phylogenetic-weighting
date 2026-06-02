from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from plm_phylo_weighting.utils.paths import model_cache_dir


def embeddings_exist(work_dir: str | Path, fam_id: str, model_key: str) -> bool:
    out_dir = model_cache_dir(work_dir, model_key)
    return (
        (out_dir / f"{fam_id}_embeddings.npy").exists()
        and (out_dir / f"{fam_id}_names.csv").exists()
    )


def save_family_embeddings(
    work_dir: str | Path,
    fam_id: str,
    sequence_names: list[str],
    embeddings: np.ndarray,
    model_key: str,
) -> None:
    out_dir = model_cache_dir(work_dir, model_key)
    np.save(out_dir / f"{fam_id}_embeddings.npy", embeddings)
    pd.Series(sequence_names, name="sequence").to_csv(out_dir / f"{fam_id}_names.csv", index=False)


def load_family_embeddings(
    work_dir: str | Path,
    fam_id: str,
    model_key: str,
) -> tuple[list[str], np.ndarray]:
    out_dir = model_cache_dir(work_dir, model_key)
    embeddings = np.load(out_dir / f"{fam_id}_embeddings.npy")
    sequence_names = pd.read_csv(out_dir / f"{fam_id}_names.csv")["sequence"].tolist()
    return sequence_names, embeddings
