from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plm_phylo_weighting.experiments.compute_embeddings import compute_all_embeddings
from plm_phylo_weighting.experiments.feature_ablation import run_feature_ablation
from plm_phylo_weighting.experiments.local_feature_signal import run_local_feature_signal_analysis
from plm_phylo_weighting.experiments.phylo_signal import run_embedding_tree_distance_analysis
from plm_phylo_weighting.experiments.prepare_data import prepare_data
from plm_phylo_weighting.experiments.rankers import run_pairwise_ranking, run_ranker_models
from plm_phylo_weighting.experiments.size_analysis import run_size_analysis
from plm_phylo_weighting.experiments.supervised import build_supervised_dataset, run_supervised
from plm_phylo_weighting.experiments.unsupervised import run_unsupervised
from plm_phylo_weighting.plots.figures import (
    plot_embedding_tree_by_size,
    plot_local_feature_signal,
    plot_model_comparison,
    plot_size_effect,
    plot_top_methods,
)
from plm_phylo_weighting.summaries.tables import combine_summary_tables
from plm_phylo_weighting.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    _, families, check_df, phylo_weights_df = prepare_data(config)
    valid_family_ids = check_df[check_df["ok"]]["FAM"].tolist()

    compute_all_embeddings(config, families, valid_family_ids)

    _, unsupervised_summary = run_unsupervised(config, families, valid_family_ids, phylo_weights_df)
    _, phylo_signal_summary, phylo_signal_size = run_embedding_tree_distance_analysis(
        config,
        families,
        valid_family_ids,
    )

    dataset_df = build_supervised_dataset(config, families, valid_family_ids, phylo_weights_df)
    local_feature_results, local_feature_summary, _ = run_local_feature_signal_analysis(config, dataset_df)

    supervised_eval, supervised_summary = run_supervised(config, dataset_df, valid_family_ids)
    pairwise_eval, pairwise_summary = run_pairwise_ranking(config, dataset_df, valid_family_ids)
    ranker_eval, ranker_summary = run_ranker_models(config, dataset_df, valid_family_ids)

    # The supervised module uses its own split, so for ablation we reproduce a deterministic split here.
    from sklearn.model_selection import train_test_split

    train_fams, test_fams = train_test_split(
        valid_family_ids,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    _, ablation_eval, ablation_summary = run_feature_ablation(
        config,
        dataset_df,
        train_fams=train_fams,
        test_fams=test_fams,
    )

    size_analysis, size_corr = run_size_analysis(
        config,
        {
            "global_regression": supervised_eval,
            "pairwise_ranking": pairwise_eval,
            "ranker": ranker_eval,
            "feature_ablation": ablation_eval,
        },
    )

    final_summary = combine_summary_tables(
        [
            unsupervised_summary,
            supervised_summary,
            pairwise_summary,
            ranker_summary,
            ablation_summary,
        ],
        [
            "unsupervised",
            "supervised",
            "pairwise",
            "ranker",
            "feature_ablation",
        ],
    )
    final_summary.to_csv(f"{config['work_dir']}/final_summary.csv", index=False)

    figure_dir = Path(config["work_dir"]) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_top_methods(final_summary, figure_dir / "top_methods.png")
    plot_model_comparison(final_summary, figure_dir / "model_comparison.png")
    plot_size_effect(size_analysis, figure_dir / "size_bins_top_methods.png")
    plot_embedding_tree_by_size(phylo_signal_size, figure_dir / "embedding_tree_distance_by_size.png")
    plot_local_feature_signal(local_feature_summary, figure_dir / "local_features_vs_phylo_weights.png")

    print("full pipeline finished")
    print(final_summary.head(30))


if __name__ == "__main__":
    main()
