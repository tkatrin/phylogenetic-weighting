from __future__ import annotations

import argparse

from plm_phylo_weighting.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()

from pathlib import Path

import pandas as pd

from plm_phylo_weighting.experiments.size_analysis import run_size_analysis


def read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    work_dir = Path(config["work_dir"])

    named_tables = {
        "global_regression": read_if_exists(work_dir / "supervised_eval.csv"),
        "pairwise_ranking": read_if_exists(work_dir / "pairwise_eval.csv"),
        "ranker": read_if_exists(work_dir / "ranker_eval.csv"),
        "feature_ablation": read_if_exists(work_dir / "feature_ablation_eval.csv"),
    }

    size_df, corr_df = run_size_analysis(config, named_tables)
    print(size_df.head(30))
    print(corr_df.head(30))


if __name__ == "__main__":
    main()
