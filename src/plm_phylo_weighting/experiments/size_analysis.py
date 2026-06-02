from __future__ import annotations

import numpy as np
import pandas as pd

from plm_phylo_weighting.evaluation.metrics import safe_corr


def add_size_bins(eval_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    if len(eval_df) == 0:
        return eval_df.copy()

    result = eval_df.copy()
    result["size_bin"] = pd.qcut(
        result["ANO"],
        q=min(n_bins, result["ANO"].nunique()),
        duplicates="drop",
    )
    return result


def summarize_by_size(eval_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()

    df = add_size_bins(eval_df)
    df["source"] = source_name

    group_cols = ["source", "model_key", "phylo_method", "method_name", "size_bin"]

    if "feature_group" in df.columns:
        group_cols.insert(4, "feature_group")

    return (
        df
        .groupby(group_cols, dropna=False, observed=False)
        .agg(
            n_families=("FAM", "count"),
            mean_ano=("ANO", "mean"),
            mean_spearman=("spearman_corr", "mean"),
            median_spearman=("spearman_corr", "median"),
            mean_pearson=("pearson_corr", "mean"),
        )
        .reset_index()
        .sort_values(["source", "mean_spearman"], ascending=[True, False])
    )


def size_correlation_table(eval_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()

    rows = []
    group_cols = ["model_key", "phylo_method", "method_name"]

    if "feature_group" in eval_df.columns:
        group_cols.append("feature_group")

    for keys, part in eval_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        spearman_size, pearson_size = safe_corr(part["ANO"], part["spearman_corr"])

        row = {
            "source": source_name,
            "size_spearman_corr": spearman_size,
            "size_pearson_corr": pearson_size,
            "n_families": len(part),
        }

        for col, value in zip(group_cols, keys):
            row[col] = value

        rows.append(row)

    return pd.DataFrame(rows)


def run_size_analysis(
    config: dict,
    named_eval_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    size_parts = []
    corr_parts = []

    for source_name, eval_df in named_eval_tables.items():
        if len(eval_df) == 0:
            continue

        size_parts.append(summarize_by_size(eval_df, source_name))
        corr_parts.append(size_correlation_table(eval_df, source_name))

    size_df = pd.concat(size_parts, ignore_index=True) if size_parts else pd.DataFrame()
    corr_df = pd.concat(corr_parts, ignore_index=True) if corr_parts else pd.DataFrame()

    size_df.to_csv(f"{config['work_dir']}/size_analysis_summary.csv", index=False)
    corr_df.to_csv(f"{config['work_dir']}/size_correlation_summary.csv", index=False)

    return size_df, corr_df
