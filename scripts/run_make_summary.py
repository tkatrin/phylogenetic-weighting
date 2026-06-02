from __future__ import annotations

import argparse

from plm_phylo_weighting.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()

from pathlib import Path

import pandas as pd

from plm_phylo_weighting.plots.figures import (
    plot_embedding_tree_by_size,
    plot_local_feature_signal,
    plot_model_comparison,
    plot_size_effect,
    plot_top_methods,
)
from plm_phylo_weighting.summaries.tables import combine_summary_tables


def read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    work_dir = Path(config["work_dir"])

    tables = [
        read_if_exists(work_dir / "unsupervised_summary.csv"),
        read_if_exists(work_dir / "supervised_summary.csv"),
        read_if_exists(work_dir / "pairwise_summary.csv"),
        read_if_exists(work_dir / "ranker_summary.csv"),
        read_if_exists(work_dir / "feature_ablation_summary.csv"),
    ]

    names = [
        "unsupervised",
        "supervised",
        "pairwise",
        "ranker",
        "feature_ablation",
    ]

    final_summary = combine_summary_tables(tables, names)
    final_summary.to_csv(work_dir / "final_summary.csv", index=False)

    figure_dir = work_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_top_methods(final_summary, figure_dir / "top_methods.png")
    plot_model_comparison(final_summary, figure_dir / "model_comparison.png")
    plot_size_effect(read_if_exists(work_dir / "size_analysis_summary.csv"), figure_dir / "size_bins_top_methods.png")
    plot_embedding_tree_by_size(read_if_exists(work_dir / "embedding_tree_distance_by_size.csv"), figure_dir / "embedding_tree_distance_by_size.png")
    plot_local_feature_signal(read_if_exists(work_dir / "density_features_vs_phylo_weights_summary.csv"), figure_dir / "local_features_vs_phylo_weights.png")

    print(final_summary.head(30))


if __name__ == "__main__":
    main()
