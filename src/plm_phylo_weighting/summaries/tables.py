from __future__ import annotations

import pandas as pd


def summarize_family_eval(eval_df: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()

    group_cols = ["model_key", "phylo_method", "method_name"]

    if "feature_group" in eval_df.columns:
        group_cols.append("feature_group")

    return (
        eval_df
        .groupby(group_cols)
        .agg(
            n_families=("FAM", "count"),
            mean_spearman=("spearman_corr", "mean"),
            median_spearman=("spearman_corr", "median"),
            mean_pearson=("pearson_corr", "mean"),
            median_pearson=("pearson_corr", "median"),
        )
        .reset_index()
        .sort_values("mean_spearman", ascending=False)
    )


def combine_summary_tables(tables: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for table, name in zip(tables, names):
        if len(table) == 0:
            continue

        part = table.copy()
        part["method_group"] = name
        parts.append(part)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True).sort_values("mean_spearman", ascending=False)
