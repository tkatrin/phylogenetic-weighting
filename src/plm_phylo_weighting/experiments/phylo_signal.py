from __future__ import annotations

import numpy as np
import pandas as pd
from ete3 import Tree
from sklearn.metrics.pairwise import cosine_distances

from plm_phylo_weighting.evaluation.metrics import safe_corr
from plm_phylo_weighting.plm.cache import load_family_embeddings
from plm_phylo_weighting.utils.cleaning import clean_array


def compute_tree_distance_matrix(newick: str, sequence_names: list[str]) -> np.ndarray:
    tree = Tree(newick, format=1)
    n_items = len(sequence_names)
    distances = np.zeros((n_items, n_items), dtype=float)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            try:
                value = tree.get_distance(sequence_names[i], sequence_names[j])
            except Exception:
                value = np.nan
            distances[i, j] = value
            distances[j, i] = value

    return distances


def compute_embedding_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    embeddings = clean_array(embeddings)
    distances = cosine_distances(embeddings)
    return np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=1.0)


def sample_upper_triangle_values(
    matrix: np.ndarray,
    max_pairs: int,
    random_state: int,
) -> np.ndarray:
    n_items = matrix.shape[0]
    i_idx, j_idx = np.triu_indices(n_items, k=1)
    total_pairs = len(i_idx)

    if total_pairs <= max_pairs:
        return matrix[i_idx, j_idx]

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(total_pairs, size=max_pairs, replace=False)
    return matrix[i_idx[chosen], j_idx[chosen]]


def run_embedding_tree_distance_analysis(
    config: dict,
    families: dict,
    valid_family_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []

    for model_key in config["model_keys_to_run"]:
        for fam_id in valid_family_ids:
            sequence_names, embeddings = load_family_embeddings(config["work_dir"], fam_id, model_key)

            tree_distances = compute_tree_distance_matrix(families[fam_id]["APH"], sequence_names)
            embedding_distances = compute_embedding_distance_matrix(embeddings)

            max_pairs = int(config.get("max_tree_distance_pairs", 200000))
            tree_values = sample_upper_triangle_values(
                tree_distances,
                max_pairs=max_pairs,
                random_state=config["random_state"],
            )
            embedding_values = sample_upper_triangle_values(
                embedding_distances,
                max_pairs=max_pairs,
                random_state=config["random_state"],
            )

            spearman_corr, pearson_corr = safe_corr(tree_values, embedding_values)

            rows.append({
                "FAM": fam_id,
                "PID": families[fam_id].get("PID"),
                "ANO": families[fam_id]["ANO"],
                "model_key": model_key,
                "comparison": "embedding_distance_vs_tree_distance",
                "spearman_corr": spearman_corr,
                "pearson_corr": pearson_corr,
                "n_pairs_used": min(len(tree_values), len(embedding_values)),
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(f"{config['work_dir']}/embedding_tree_distance_correlation.csv", index=False)

    summary_df = (
        results_df
        .groupby(["model_key", "comparison"])
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
    summary_df.to_csv(
        f"{config['work_dir']}/embedding_tree_distance_correlation_summary.csv",
        index=False,
    )

    size_summary_df = summarize_embedding_tree_distance_by_size(results_df)
    size_summary_df.to_csv(
        f"{config['work_dir']}/embedding_tree_distance_by_size.csv",
        index=False,
    )

    return results_df, summary_df, size_summary_df


def summarize_embedding_tree_distance_by_size(
    results_df: pd.DataFrame,
    n_bins: int = 5,
) -> pd.DataFrame:
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
        .groupby(["model_key", "comparison", "size_bin"], observed=False)
        .agg(
            n_families=("FAM", "count"),
            mean_ano=("ANO", "mean"),
            mean_spearman=("spearman_corr", "mean"),
            median_spearman=("spearman_corr", "median"),
            mean_pearson=("pearson_corr", "mean"),
            median_pearson=("pearson_corr", "median"),
        )
        .reset_index()
    )
