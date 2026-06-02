from __future__ import annotations

import argparse

from plm_phylo_weighting.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()

from plm_phylo_weighting.experiments.prepare_data import prepare_data
from plm_phylo_weighting.experiments.supervised import build_supervised_dataset
from plm_phylo_weighting.experiments.rankers import run_pairwise_ranking, run_ranker_models


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, families, check_df, phylo_weights_df = prepare_data(config)
    valid_family_ids = check_df[check_df["ok"]]["FAM"].tolist()
    dataset_df = build_supervised_dataset(config, families, valid_family_ids, phylo_weights_df)

    _, pairwise_summary = run_pairwise_ranking(config, dataset_df, valid_family_ids)
    _, ranker_summary = run_ranker_models(config, dataset_df, valid_family_ids)

    print("pairwise")
    print(pairwise_summary.head(20))
    print("rankers")
    print(ranker_summary.head(20))


if __name__ == "__main__":
    main()
