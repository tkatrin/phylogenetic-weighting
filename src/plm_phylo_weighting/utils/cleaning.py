from __future__ import annotations

import numpy as np
import pandas as pd


def clean_array(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.nan_to_num(array, nan=fill_value, posinf=fill_value, neginf=fill_value)


def clean_numeric_frame(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    result = df.copy()
    result = result.replace([np.inf, -np.inf], np.nan)

    for col in result.columns:
        if pd.api.types.is_numeric_dtype(result[col]):
            median = result[col].median()
            if pd.isna(median):
                median = fill_value
            result[col] = result[col].fillna(median)

    return result


def safe_normalize_array(values: np.ndarray, normalize_sum: float) -> np.ndarray:
    weights = clean_array(values)
    weights[weights < 0] = 0.0
    total = float(weights.sum())

    if len(weights) == 0:
        return weights

    if not np.isfinite(total) or total <= 1e-12:
        return np.ones_like(weights, dtype=float) * (normalize_sum / len(weights))

    return weights * normalize_sum / total


def safe_normalize_dict(weights: dict[str, float], normalize_sum: float) -> dict[str, float]:
    total = float(sum(weights.values()))

    if len(weights) == 0:
        return {}

    if not np.isfinite(total) or total <= 1e-12:
        value = normalize_sum / len(weights)
        return {key: value for key in weights}

    return {key: value * normalize_sum / total for key, value in weights.items()}
