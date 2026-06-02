from __future__ import annotations

import pandas as pd

from plm_phylo_weighting.evaluation.metrics import safe_corr


DEFAULT_LOCAL_FEATURES = [
    "mean_distance_all",
    "nearest_1",
    "nearest_2",
    "mean_knn_3",
    "mean_knn_5",
    "mean_knn_10",
    "kernel_density_5",
    "inv_kernel_density_5",
    "mean_distance_all_rank",
    "nearest_1_rank",
    "mean_knn_3_rank",
    "mean_knn_5_rank",
    "mean_knn_10_rank",
    "kernel_density_5_rank",
    "inv_kernel_density_5_rank",
]


def run_local_feature_signal_analysis(
    config: dict,
    dataset_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = [col for col in DEFAULT_LOCAL_FEATURES if col in dataset_df.columns]
    rows = []

    for model_key in config["model_keys_to_run"]:
        for phylo_method in config["phylo_methods"]:
            subset = dataset_df[
                (dataset_df["model_key"] == model_key)
                & (dataset_df["phylo_method"] == phylo_method)
            ].copy()

            for fam_id, part in subset.groupby("FAM"):
                for feature in feature_cols:
                    spearman_corr, pearson_corr = safe_corr(part["phylo_weight"], part[feature])

                    rows.append({
                        "FAM": fam_id,
                        "PID": part["PID"].iloc[0],
                        "ANO": part["ANO"].iloc[0],
                        "model_key": model_key,
                        "phylo_method": phylo_method,
                        "feature": feature,
                        "spearman_corr": spearman_corr,
                        "pearson_corr": pearson_corr,
                        "n_sequences": len(part),
                    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{config['work_dir']}/density_features_vs_phylo_weights.csv", index=False)

    summary_df = summarize_local_feature_signal(results_df)
    summary_df.to_csv(
        f"{config['work_dir']}/density_features_vs_phylo_weights_summary.csv",
        index=False,
    )

    size_summary_df = summarize_local_feature_signal_by_size(results_df)
    size_summary_df.to_csv(
        f"{config['work_dir']}/density_features_vs_phylo_weights_by_size.csv",
        index=False,
    )

    return results_df, summary_df, size_summary_df


def summarize_local_feature_signal(results_df: pd.DataFrame) -> pd.DataFrame:
    if len(results_df) == 0:
        return pd.DataFrame()

    return (
        results_df
        .groupby(["model_key", "phylo_method", "feature"])
        .agg(
            n_families=("FAM", "count"),
            mean_spearman=("spearman_corr", "mean"),
            median_spearman=("spearman_corr", "median"),
            mean_pearson=("pearson_corr", "mean"),
            median_pearson=("pearson_corr", "median"),
            mean_ano=("ANO", "mean"),
        )
        .reset_index()
        .sort_values("mean_spearman", ascending=False)
    )


def summarize_local_feature_signal_by_size(results_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    if len(results_df) == 0:
        return pd.DataFrame()

    df = results_df.copy()
    df["size_bin"] = pd.qcut(
        df["ANO"],
        q=min(n_bins, df["ANO"].nunique()),
        duplicates="drop",
    )

    return (
        df
        .groupby(["model_key", "phylo_method", "feature", "size_bin"], observed=False)
        .agg(
            n_families=("FAM", "count"),
            mean_ano=("ANO", "mean"),
            mean_spearman=("spearman_corr", "mean"),
            median_spearman=("spearman_corr", "median"),
            mean_pearson=("pearson_corr", "mean"),
            median_pearson=("pearson_corr", "median"),
        )
        .reset_index()
        .sort_values(["model_key", "phylo_method", "mean_spearman"], ascending=[True, True, False])
    )
