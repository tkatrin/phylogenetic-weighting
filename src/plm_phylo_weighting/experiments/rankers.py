from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from plm_phylo_weighting.evaluation.metrics import evaluate_family_predictions
from plm_phylo_weighting.models.features import get_feature_cols
from plm_phylo_weighting.models.pairwise import (
    make_pairwise_dataset,
    make_pairwise_model,
    score_family_pairwise,
)
from plm_phylo_weighting.models.rankers import fit_predict_lightgbm_ranker, fit_predict_xgboost_ranker
from plm_phylo_weighting.summaries.tables import summarize_family_eval


def run_pairwise_ranking(config: dict, dataset_df: pd.DataFrame, valid_family_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fams, test_fams = train_test_split(
        valid_family_ids,
        test_size=config["test_size"],
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

            x_pair, y_pair = make_pairwise_dataset(
                df=train_df,
                fams=train_fams,
                feature_cols=feature_cols,
                max_pairs_per_family=config["max_pairs_per_family"],
                random_state=config["random_state"],
            )

            if len(y_pair) == 0 or len(set(y_pair)) < 2:
                continue

            model = make_pairwise_model()
            model.fit(x_pair, y_pair)

            for fam_id in test_fams:
                fam_df = test_df[test_df["FAM"] == fam_id].copy()
                if len(fam_df) == 0:
                    continue

                scores = score_family_pairwise(model, fam_df, feature_cols)
                pred_df = fam_df[["FAM", "ANO", "sequence", "phylo_weight"]].copy()
                pred_df["prediction"] = scores
                pred_df["method_name"] = "pairwise_logistic_ranking"
                pred_df["model_key"] = model_key
                pred_df["phylo_method"] = phylo_method
                rows.append(pred_df)

    predictions_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    eval_df = evaluate_family_predictions(predictions_df) if len(predictions_df) else pd.DataFrame()
    summary_df = summarize_family_eval(eval_df)

    predictions_df.to_csv(f"{config['work_dir']}/pairwise_predictions.csv", index=False)
    eval_df.to_csv(f"{config['work_dir']}/pairwise_eval.csv", index=False)
    summary_df.to_csv(f"{config['work_dir']}/pairwise_summary.csv", index=False)

    return eval_df, summary_df


def run_ranker_models(config: dict, dataset_df: pd.DataFrame, valid_family_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fams, test_fams = train_test_split(
        valid_family_ids,
        test_size=config["test_size"],
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

            for runner in [fit_predict_lightgbm_ranker, fit_predict_xgboost_ranker]:
                pred_df = runner(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=feature_cols,
                    target_col="phylo_weight",
                    random_state=config["random_state"],
                )

                if pred_df is None:
                    continue

                pred_df["model_key"] = model_key
                pred_df["phylo_method"] = phylo_method
                rows.append(pred_df)

    predictions_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    eval_df = evaluate_family_predictions(predictions_df) if len(predictions_df) else pd.DataFrame()
    summary_df = summarize_family_eval(eval_df)

    predictions_df.to_csv(f"{config['work_dir']}/ranker_predictions.csv", index=False)
    eval_df.to_csv(f"{config['work_dir']}/ranker_eval.csv", index=False)
    summary_df.to_csv(f"{config['work_dir']}/ranker_summary.csv", index=False)

    return eval_df, summary_df
