from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from plm_phylo_weighting.evaluation.metrics import evaluate_family_predictions
from plm_phylo_weighting.models.features import build_pca_features, compute_local_features, get_feature_cols
from plm_phylo_weighting.models.regression import fit_predict_regression
from plm_phylo_weighting.plm.cache import load_family_embeddings
from plm_phylo_weighting.summaries.tables import summarize_family_eval
from plm_phylo_weighting.utils.cleaning import clean_numeric_frame


def build_supervised_dataset(config: dict, families: dict, valid_family_ids: list[str], phylo_weights_df: pd.DataFrame) -> pd.DataFrame:
    feature_parts = []

    for model_key in config["model_keys_to_run"]:
        for fam_id in valid_family_ids:
            sequence_names, embeddings = load_family_embeddings(config["work_dir"], fam_id, model_key)

            local_df = compute_local_features(
                fam_id=fam_id,
                pid=families[fam_id].get("PID"),
                ano=families[fam_id]["ANO"],
                sequence_names=sequence_names,
                embeddings=embeddings,
                model_key=model_key,
            )

            pca_df = build_pca_features(
                embeddings=embeddings,
                n_components=config["pca_components"],
                index=local_df.index,
            )

            feature_parts.append(pd.concat([local_df, pca_df], axis=1))

    features_df = pd.concat(feature_parts, ignore_index=True)
    features_df = clean_numeric_frame(features_df)
    features_df.to_csv(f"{config['work_dir']}/features.csv", index=False)

    dataset_df = features_df.merge(
        phylo_weights_df,
        on=["FAM", "PID", "ANO", "sequence"],
        how="inner",
    )
    dataset_df.to_csv(f"{config['work_dir']}/supervised_dataset.csv", index=False)
    return dataset_df


def run_supervised(config: dict, dataset_df: pd.DataFrame, valid_family_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fams, test_fams = train_test_split(
        valid_family_ids,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    train_fams, _ = train_test_split(
        train_fams,
        test_size=config["valid_size"],
        random_state=config["random_state"],
    )

    rows = []

    for model_key in config["model_keys_to_run"]:
        for phylo_method in config["phylo_methods"]:
            subset = dataset_df[
                (dataset_df["model_key"] == model_key)
                & (dataset_df["phylo_method"] == phylo_method)
            ].copy()

            feature_cols = get_feature_cols(subset)

            train_df = subset[subset["FAM"].isin(train_fams)]
            test_df = subset[subset["FAM"].isin(test_fams)]

            pred_df = fit_predict_regression(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                target_col="phylo_weight",
                random_state=config["random_state"],
            )
            pred_df["model_key"] = model_key
            pred_df["phylo_method"] = phylo_method
            rows.append(pred_df)

    predictions_df = pd.concat(rows, ignore_index=True)
    eval_df = evaluate_family_predictions(predictions_df)
    summary_df = summarize_family_eval(eval_df)

    predictions_df.to_csv(f"{config['work_dir']}/supervised_predictions.csv", index=False)
    eval_df.to_csv(f"{config['work_dir']}/supervised_eval.csv", index=False)
    summary_df.to_csv(f"{config['work_dir']}/supervised_summary.csv", index=False)

    return eval_df, summary_df
