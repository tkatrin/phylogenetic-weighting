from __future__ import annotations

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from plm_phylo_weighting.evaluation.metrics import evaluate_family_predictions
from plm_phylo_weighting.models.features import get_feature_cols
from plm_phylo_weighting.utils.cleaning import clean_numeric_frame


def get_ablation_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    pca_cols = [col for col in df.columns if col.startswith("pca_")]

    density_cols = [
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
    ]

    rank_cols = [
        "mean_distance_all_rank",
        "nearest_1_rank",
        "mean_knn_3_rank",
        "mean_knn_5_rank",
        "mean_knn_10_rank",
        "kernel_density_5_rank",
        "inv_kernel_density_5_rank",
    ]

    groups = {
        "pca_only": pca_cols,
        "density_only": density_cols,
        "rank_only": ["ANO"] + rank_cols,
        "density_plus_rank": density_cols + rank_cols,
        "pca_plus_density": pca_cols + density_cols,
        "pca_plus_density_plus_rank": pca_cols + density_cols + rank_cols,
        "all_features": get_feature_cols(df),
    }

    return {
        name: [col for col in cols if col in df.columns]
        for name, cols in groups.items()
        if len([col for col in cols if col in df.columns]) > 0
    }


def run_feature_ablation(
    config: dict,
    dataset_df: pd.DataFrame,
    train_fams: list[str],
    test_fams: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts = []

    for model_key in config["model_keys_to_run"]:
        for phylo_method in config["phylo_methods"]:
            subset = dataset_df[
                (dataset_df["model_key"] == model_key)
                & (dataset_df["phylo_method"] == phylo_method)
            ].copy()

            train_df = subset[subset["FAM"].isin(train_fams)].copy()
            test_df = subset[subset["FAM"].isin(test_fams)].copy()

            for feature_group, feature_cols in get_ablation_feature_groups(subset).items():
                model = ExtraTreesRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    random_state=config["random_state"],
                    n_jobs=-1,
                )

                model.fit(
                    clean_numeric_frame(train_df[feature_cols]),
                    train_df["phylo_weight"].to_numpy(),
                )

                pred_df = test_df[["FAM", "ANO", "sequence", "phylo_weight"]].copy()
                pred_df["prediction"] = model.predict(clean_numeric_frame(test_df[feature_cols]))
                pred_df["method_name"] = "extra_trees_ablation"
                pred_df["feature_group"] = feature_group
                pred_df["model_key"] = model_key
                pred_df["phylo_method"] = phylo_method
                prediction_parts.append(pred_df)

    predictions_df = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    predictions_df.to_csv(f"{config['work_dir']}/feature_ablation_predictions.csv", index=False)

    if len(predictions_df) == 0:
        eval_df = pd.DataFrame()
        summary_df = pd.DataFrame()
    else:
        eval_df = evaluate_family_predictions_with_feature_group(predictions_df)
        summary_df = summarize_feature_ablation(eval_df)

    eval_df.to_csv(f"{config['work_dir']}/feature_ablation_eval.csv", index=False)
    summary_df.to_csv(f"{config['work_dir']}/feature_ablation_summary.csv", index=False)

    return predictions_df, eval_df, summary_df


def evaluate_family_predictions_with_feature_group(predictions_df: pd.DataFrame) -> pd.DataFrame:
    from plm_phylo_weighting.evaluation.metrics import safe_corr

    rows = []

    for keys, part in predictions_df.groupby(
        ["model_key", "phylo_method", "method_name", "feature_group", "FAM"]
    ):
        model_key, phylo_method, method_name, feature_group, fam_id = keys
        spearman_corr, pearson_corr = safe_corr(part["phylo_weight"], part["prediction"])

        rows.append({
            "model_key": model_key,
            "phylo_method": phylo_method,
            "method_name": method_name,
            "feature_group": feature_group,
            "FAM": fam_id,
            "ANO": int(part["ANO"].iloc[0]),
            "spearman_corr": spearman_corr,
            "pearson_corr": pearson_corr,
            "n_sequences": len(part),
        })

    return pd.DataFrame(rows)


def summarize_feature_ablation(eval_df: pd.DataFrame) -> pd.DataFrame:
    if len(eval_df) == 0:
        return pd.DataFrame()

    return (
        eval_df
        .groupby(["model_key", "phylo_method", "method_name", "feature_group"])
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
