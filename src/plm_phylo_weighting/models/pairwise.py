from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plm_phylo_weighting.models.regression import clean_features


def make_pairwise_dataset(
    df: pd.DataFrame,
    fams: list[str],
    feature_cols: list[str],
    max_pairs_per_family: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for fam_id in fams:
        fam_df = df[df["FAM"] == fam_id].reset_index(drop=True)
        n_items = len(fam_df)

        if n_items < 2:
            continue

        max_possible = n_items * (n_items - 1) // 2
        n_pairs = min(max_pairs_per_family, max_possible)

        pairs: set[tuple[int, int]] = set()
        attempts = 0

        while len(pairs) < n_pairs and attempts < n_pairs * 20:
            i = int(rng.integers(0, n_items))
            j = int(rng.integers(0, n_items))
            attempts += 1

            if i == j:
                continue

            pairs.add(tuple(sorted((i, j))))

        if not pairs:
            continue

        x = clean_features(fam_df, feature_cols).to_numpy()
        y = fam_df["phylo_weight"].to_numpy()

        rows: list[np.ndarray] = []
        labels: list[int] = []

        for i, j in pairs:
            diff = x[i] - x[j]
            rows.append(np.concatenate([diff, np.abs(diff)]))
            labels.append(1 if y[i] > y[j] else 0)

        x_parts.append(np.vstack(rows))
        y_parts.append(np.asarray(labels))

    if not x_parts:
        return np.empty((0, 2 * len(feature_cols))), np.empty((0,))

    return np.vstack(x_parts), np.concatenate(y_parts)


def make_pairwise_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1),
    )


def score_family_pairwise(model, fam_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    x = clean_features(fam_df, feature_cols).to_numpy()
    n_items = len(fam_df)
    scores = np.zeros(n_items)

    for i in range(n_items):
        rows: list[np.ndarray] = []

        for j in range(n_items):
            if i == j:
                continue

            diff = x[i] - x[j]
            rows.append(np.concatenate([diff, np.abs(diff)]))

        if rows:
            scores[i] = model.predict_proba(np.vstack(rows))[:, 1].mean()

    return scores
