from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

from plm_phylo_weighting.utils.cleaning import clean_array, clean_numeric_frame


LOCAL_FEATURE_COLS = [
    "ANO",
    "mean_distance_all",
    "std_distance_all",
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


def rank_normalized(values: np.ndarray) -> np.ndarray:
    values = clean_array(values)
    order = pd.Series(values).rank(method="average").to_numpy()

    if len(values) <= 1:
        return np.zeros_like(values)

    return (order - 1) / (len(values) - 1)


def compute_local_features(
    fam_id: str,
    pid: str | None,
    ano: int,
    sequence_names: list[str],
    embeddings: np.ndarray,
    model_key: str,
) -> pd.DataFrame:
    embeddings = clean_array(embeddings)
    n_items = len(sequence_names)

    if n_items <= 1:
        base = np.zeros(n_items, dtype=float)
        features = pd.DataFrame({
            "FAM": fam_id,
            "PID": pid,
            "ANO": ano,
            "sequence": sequence_names,
            "model_key": model_key,
            "mean_distance_all": base,
            "std_distance_all": base,
            "nearest_1": base,
            "nearest_2": base,
            "mean_knn_3": base,
            "mean_knn_5": base,
            "mean_knn_10": base,
            "kernel_density_5": np.ones(n_items),
            "inv_kernel_density_5": np.ones(n_items),
        })
    else:
        distances = cosine_distances(embeddings)
        distances = np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=1.0)

        nearest_matrix = distances.copy()
        np.fill_diagonal(nearest_matrix, np.inf)
        sorted_dist = np.sort(nearest_matrix, axis=1)
        finite_sorted = clean_array(sorted_dist[:, :min(20, n_items - 1)])

        all_distances = distances.copy()
        np.fill_diagonal(all_distances, np.nan)

        sigma = 0.005
        k = min(5, n_items - 1)
        density_dist = finite_sorted[:, :k]
        kernel_density = np.exp(-(density_dist ** 2) / (2 * sigma ** 2)).sum(axis=1)

        features = pd.DataFrame({
            "FAM": fam_id,
            "PID": pid,
            "ANO": ano,
            "sequence": sequence_names,
            "model_key": model_key,
            "mean_distance_all": np.nanmean(all_distances, axis=1),
            "std_distance_all": np.nanstd(all_distances, axis=1),
            "nearest_1": finite_sorted[:, 0],
            "nearest_2": finite_sorted[:, min(1, finite_sorted.shape[1] - 1)],
            "mean_knn_3": finite_sorted[:, :min(3, finite_sorted.shape[1])].mean(axis=1),
            "mean_knn_5": finite_sorted[:, :min(5, finite_sorted.shape[1])].mean(axis=1),
            "mean_knn_10": finite_sorted[:, :min(10, finite_sorted.shape[1])].mean(axis=1),
            "kernel_density_5": kernel_density,
            "inv_kernel_density_5": 1.0 / (kernel_density + 1e-8),
        })

    for col in [
        "mean_distance_all",
        "nearest_1",
        "mean_knn_3",
        "mean_knn_5",
        "mean_knn_10",
        "kernel_density_5",
        "inv_kernel_density_5",
    ]:
        features[col + "_rank"] = rank_normalized(features[col].to_numpy())

    return clean_numeric_frame(features)


def build_pca_features(
    embeddings: np.ndarray,
    n_components: int,
    index: pd.Index,
    prefix: str = "pca",
) -> pd.DataFrame:
    embeddings = clean_array(embeddings)

    if embeddings.shape[0] < 2:
        return pd.DataFrame(index=index)

    n_components = min(n_components, embeddings.shape[0] - 1, embeddings.shape[1])

    if n_components < 1:
        return pd.DataFrame(index=index)

    scaled = StandardScaler().fit_transform(embeddings)
    scaled = clean_array(scaled)

    pca = PCA(n_components=n_components, random_state=42)
    values = clean_array(pca.fit_transform(scaled))

    return pd.DataFrame(
        values,
        columns=[f"{prefix}_{i}" for i in range(n_components)],
        index=index,
    )


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    pca_cols = [col for col in df.columns if col.startswith("pca_")]
    return LOCAL_FEATURE_COLS + pca_cols
