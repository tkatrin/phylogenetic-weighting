from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from plm_phylo_weighting.utils.cleaning import clean_array, safe_normalize_array


def compute_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    clean_embeddings = clean_array(embeddings)
    distances = cosine_distances(clean_embeddings)
    return np.nan_to_num(distances, nan=0.0, posinf=1.0, neginf=1.0)


def compute_raw_mean_distance_weights(
    embeddings: np.ndarray,
    sequence_names: list[str],
    normalize_sum: float | None = None,
) -> dict[str, float]:
    distances = compute_distance_matrix(embeddings)
    np.fill_diagonal(distances, np.nan)
    raw_weights = clean_array(np.nanmean(distances, axis=1))

    if normalize_sum is not None:
        raw_weights = safe_normalize_array(raw_weights, normalize_sum)

    return dict(zip(sequence_names, raw_weights))


def compute_knn_mean_distance_weights(
    embeddings: np.ndarray,
    sequence_names: list[str],
    k: int,
    normalize_sum: float | None = None,
) -> dict[str, float]:
    distances = compute_distance_matrix(embeddings)
    np.fill_diagonal(distances, np.inf)

    n_items = len(sequence_names)

    if n_items <= 1:
        raw_weights = np.ones(n_items, dtype=float)
    else:
        k = max(1, min(k, n_items - 1))
        nearest = np.sort(distances, axis=1)[:, :k]
        raw_weights = clean_array(nearest.mean(axis=1))

    if normalize_sum is not None:
        raw_weights = safe_normalize_array(raw_weights, normalize_sum)

    return dict(zip(sequence_names, raw_weights))


def compute_kernel_density_weights(
    embeddings: np.ndarray,
    sequence_names: list[str],
    k: int,
    sigma: float,
    normalize_sum: float | None = None,
) -> dict[str, float]:
    distances = compute_distance_matrix(embeddings)
    np.fill_diagonal(distances, np.inf)

    n_items = len(sequence_names)
    sigma = max(float(sigma), 1e-6)

    if n_items <= 1:
        raw_weights = np.ones(n_items, dtype=float)
    else:
        k = max(1, min(k, n_items - 1))
        nearest = np.sort(distances, axis=1)[:, :k]
        density = np.exp(-(nearest ** 2) / (2 * sigma ** 2)).sum(axis=1)
        raw_weights = 1.0 / (clean_array(density) + 1e-8)

    if normalize_sum is not None:
        raw_weights = safe_normalize_array(raw_weights, normalize_sum)

    return dict(zip(sequence_names, raw_weights))


def compute_graph_density_weights(
    embeddings: np.ndarray,
    sequence_names: list[str],
    k: int,
    sigma: float,
    normalize_sum: float | None = None,
) -> dict[str, float]:
    distances = compute_distance_matrix(embeddings)
    np.fill_diagonal(distances, np.inf)

    n_items = len(sequence_names)
    sigma = max(float(sigma), 1e-6)

    if n_items <= 1:
        raw_weights = np.ones(n_items, dtype=float)
    else:
        k = max(1, min(k, n_items - 1))
        nearest_idx = np.argsort(distances, axis=1)[:, :k]
        density = np.zeros(n_items, dtype=float)

        for i in range(n_items):
            for j in nearest_idx[i]:
                sim = np.exp(-(distances[i, j] ** 2) / (2 * sigma ** 2))
                if np.isfinite(sim):
                    density[i] += sim
                    density[j] += sim

        raw_weights = 1.0 / (clean_array(density) + 1e-8)

    if normalize_sum is not None:
        raw_weights = safe_normalize_array(raw_weights, normalize_sum)

    return dict(zip(sequence_names, raw_weights))
