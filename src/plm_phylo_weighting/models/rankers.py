from __future__ import annotations

import pandas as pd

from plm_phylo_weighting.models.regression import clean_features


def fit_predict_lightgbm_ranker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "phylo_weight",
    random_state: int = 42,
) -> pd.DataFrame | None:
    try:
        import lightgbm as lgb
    except Exception:
        return None

    train_df = train_df.sort_values("FAM").reset_index(drop=True)
    group_sizes = train_df.groupby("FAM").size().tolist()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        clean_features(train_df, feature_cols),
        train_df[target_col].to_numpy(),
        group=group_sizes,
    )

    pred_df = test_df[["FAM", "ANO", "sequence", target_col]].copy()
    pred_df["method_name"] = "lightgbm_ranker"
    pred_df["prediction"] = model.predict(clean_features(test_df, feature_cols))
    return pred_df


def fit_predict_xgboost_ranker(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "phylo_weight",
    random_state: int = 42,
) -> pd.DataFrame | None:
    try:
        import xgboost as xgb
    except Exception:
        return None

    train_df = train_df.sort_values("FAM").reset_index(drop=True)
    group_sizes = train_df.groupby("FAM").size().tolist()

    model = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    model.fit(
        clean_features(train_df, feature_cols),
        train_df[target_col].to_numpy(),
        group=group_sizes,
        verbose=False,
    )

    pred_df = test_df[["FAM", "ANO", "sequence", target_col]].copy()
    pred_df["method_name"] = "xgboost_ranker"
    pred_df["prediction"] = model.predict(clean_features(test_df, feature_cols))
    return pred_df
