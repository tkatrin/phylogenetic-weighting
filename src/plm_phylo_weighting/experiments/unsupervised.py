from __future__ import annotations

import pandas as pd

from plm_phylo_weighting.evaluation.metrics import safe_corr
from plm_phylo_weighting.plm.cache import load_family_embeddings
from plm_phylo_weighting.summaries.tables import summarize_family_eval
from plm_phylo_weighting.weights.plm_density import (
    compute_graph_density_weights,
    compute_kernel_density_weights,
    compute_knn_mean_distance_weights,
    compute_raw_mean_distance_weights,
)


def evaluate_weights(target_df: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float, int]:
    pred_df = pd.DataFrame({"sequence": list(weights.keys()), "prediction": list(weights.values())})
    merged = target_df.merge(pred_df, on="sequence")
    spearman, pearson = safe_corr(merged["phylo_weight"], merged["prediction"])
    return spearman, pearson, len(merged)


def run_unsupervised(config: dict, families: dict, valid_family_ids: list[str], phylo_weights_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    for model_key in config["model_keys_to_run"]:
        for fam_id in valid_family_ids:
            sequence_names, embeddings = load_family_embeddings(config["work_dir"], fam_id, model_key)
            normalize_sum = families[fam_id]["ANO"]

            methods = [
                ("raw_mean_distance", {}, compute_raw_mean_distance_weights(embeddings, sequence_names, normalize_sum)),
            ]

            for k in config["knn_values"]:
                methods.append((
                    "knn_mean_distance",
                    {"k": k},
                    compute_knn_mean_distance_weights(embeddings, sequence_names, k, normalize_sum),
                ))

            for k in config["density_k_values"]:
                for sigma in config["sigma_values"]:
                    methods.append((
                        "kernel_density",
                        {"k": k, "sigma": sigma},
                        compute_kernel_density_weights(embeddings, sequence_names, k, sigma, normalize_sum),
                    ))
                    methods.append((
                        "graph_density",
                        {"k": k, "sigma": sigma},
                        compute_graph_density_weights(embeddings, sequence_names, k, sigma, normalize_sum),
                    ))

            for method_name, params, weights in methods:
                for phylo_method in config["phylo_methods"]:
                    target_df = phylo_weights_df[
                        (phylo_weights_df["FAM"] == fam_id)
                        & (phylo_weights_df["phylo_method"] == phylo_method)
                    ]
                    spearman, pearson, n_sequences = evaluate_weights(target_df, weights)

                    rows.append({
                        "model_key": model_key,
                        "phylo_method": phylo_method,
                        "method_name": method_name,
                        "FAM": fam_id,
                        "ANO": families[fam_id]["ANO"],
                        "spearman_corr": spearman,
                        "pearson_corr": pearson,
                        "n_sequences": n_sequences,
                        **params,
                    })

    results_df = pd.DataFrame(rows)
    summary_df = summarize_family_eval(results_df)

    results_df.to_csv(f"{config['work_dir']}/unsupervised_results.csv", index=False)
    summary_df.to_csv(f"{config['work_dir']}/unsupervised_summary.csv", index=False)

    return results_df, summary_df
