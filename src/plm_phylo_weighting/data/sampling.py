from __future__ import annotations

import numpy as np
import pandas as pd


def select_stratified_families(
    summary_df: pd.DataFrame,
    n_families: int,
    min_ano: int,
    random_state: int = 42,
) -> pd.DataFrame:
    candidate_df = summary_df[
        (summary_df["ANO"].fillna(0) >= min_ano)
        & (summary_df["has_APH"])
    ].copy()

    candidate_df = candidate_df.sort_values("ANO").reset_index(drop=True)
    candidate_df["size_bin"] = pd.qcut(
        candidate_df["ANO"],
        q=min(5, len(candidate_df)),
        labels=False,
        duplicates="drop",
    )

    selected_parts: list[pd.DataFrame] = []
    n_bins = int(candidate_df["size_bin"].nunique())
    per_bin = max(1, n_families // n_bins)
    rng = np.random.default_rng(random_state)

    for _, part in candidate_df.groupby("size_bin"):
        take = min(per_bin, len(part))
        sampled_idx = rng.choice(part.index.to_numpy(), size=take, replace=False)
        selected_parts.append(candidate_df.loc[sampled_idx])

    selected_df = pd.concat(selected_parts, ignore_index=True)

    if len(selected_df) < n_families:
        remaining = candidate_df[~candidate_df["FAM"].isin(selected_df["FAM"])]
        take = min(n_families - len(selected_df), len(remaining))
        extra_idx = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
        selected_df = pd.concat([selected_df, candidate_df.loc[extra_idx]], ignore_index=True)

    return selected_df.sample(frac=1.0, random_state=random_state).head(n_families).reset_index(drop=True)


def select_target_mean_families(
    summary_df: pd.DataFrame,
    n_families: int,
    min_ano: int,
    target_mean_ano: float,
    random_state: int = 42,
) -> pd.DataFrame:
    candidate_df = summary_df[
        (summary_df["ANO"].fillna(0) >= min_ano)
        & (summary_df["has_APH"])
    ].copy()

    candidate_df = candidate_df.sort_values("ANO").reset_index(drop=True)
    candidate_df["size_bin"] = pd.qcut(
        candidate_df["ANO"],
        q=min(8, len(candidate_df)),
        labels=False,
        duplicates="drop",
    )

    selected_parts: list[pd.DataFrame] = []
    n_bins = int(candidate_df["size_bin"].nunique())
    per_bin = max(1, n_families // n_bins)
    rng = np.random.default_rng(random_state)

    for _, part in candidate_df.groupby("size_bin"):
        take = min(per_bin, len(part))
        sampled_idx = rng.choice(part.index.to_numpy(), size=take, replace=False)
        selected_parts.append(candidate_df.loc[sampled_idx])

    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df = selected_df.drop_duplicates("FAM").reset_index(drop=True)

    if len(selected_df) < n_families:
        remaining = candidate_df[~candidate_df["FAM"].isin(selected_df["FAM"])]
        take = min(n_families - len(selected_df), len(remaining))
        extra_idx = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
        selected_df = pd.concat([selected_df, candidate_df.loc[extra_idx]], ignore_index=True)

    selected_df = selected_df.head(n_families).copy()

    for _ in range(1000):
        if selected_df["ANO"].mean() >= target_mean_ano:
            break

        selected_small_idx = selected_df["ANO"].idxmin()
        remaining = candidate_df[~candidate_df["FAM"].isin(selected_df["FAM"])]

        if len(remaining) == 0:
            break

        larger_remaining = remaining[remaining["ANO"] > selected_df.loc[selected_small_idx, "ANO"]]

        if len(larger_remaining) == 0:
            break

        replacement = larger_remaining.sort_values("ANO", ascending=False).head(1)
        selected_df = selected_df.drop(index=selected_small_idx)
        selected_df = pd.concat([selected_df, replacement], ignore_index=True)

    selected_df = selected_df.sample(frac=1.0, random_state=random_state).head(n_families).reset_index(drop=True)
    selected_df["size_bin"] = pd.qcut(
        selected_df["ANO"],
        q=min(5, len(selected_df)),
        labels=False,
        duplicates="drop",
    )

    return selected_df
