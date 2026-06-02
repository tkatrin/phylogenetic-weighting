from __future__ import annotations

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from plm_phylo_weighting.utils.cleaning import clean_numeric_frame


def clean_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    return clean_numeric_frame(df[feature_cols])


def make_regression_models(random_state: int = 42):
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.1,
            random_state=random_state,
        ),
    }


def fit_predict_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "phylo_weight",
    random_state: int = 42,
) -> pd.DataFrame:
    x_train = clean_features(train_df, feature_cols)
    y_train = train_df[target_col].to_numpy()

    rows: list[pd.DataFrame] = []

    for method_name, model in make_regression_models(random_state).items():
        model.fit(x_train, y_train)

        pred_df = test_df[["FAM", "ANO", "sequence", target_col]].copy()
        pred_df["method_name"] = method_name
        pred_df["prediction"] = model.predict(clean_features(test_df, feature_cols))
        rows.append(pred_df)

    return pd.concat(rows, ignore_index=True)
