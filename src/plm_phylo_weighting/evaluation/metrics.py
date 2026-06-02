from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from plm_phylo_weighting.utils.cleaning import clean_array


def safe_corr(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> tuple[float, float]:
    first = clean_array(np.asarray(x, dtype=float))
    second = clean_array(np.asarray(y, dtype=float))

    mask = np.isfinite(first) & np.isfinite(second)
    first = first[mask]
    second = second[mask]

    if len(first) < 3:
        return np.nan, np.nan

    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return np.nan, np.nan

    return float(spearmanr(first, second)[0]), float(pearsonr(first, second)[0])


def evaluate_family_predictions(
    pred_df: pd.DataFrame,
    target_col: str = "phylo_weight",
    pred_col: str = "prediction",
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []

    for keys, part in pred_df.groupby(["model_key", "phylo_method", "method_name", "FAM"]):
        model_key, phylo_method, method_name, fam_id = keys
        spearman_corr, pearson_corr = safe_corr(part[target_col], part[pred_col])
        rows.append({
            "model_key": model_key,
            "phylo_method": phylo_method,
            "method_name": method_name,
            "FAM": fam_id,
            "ANO": int(part["ANO"].iloc[0]),
            "spearman_corr": spearman_corr,
            "pearson_corr": pearson_corr,
            "n_sequences": int(len(part)),
        })

    return pd.DataFrame(rows)
