from __future__ import annotations

import argparse

from plm_phylo_weighting.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()

from plm_phylo_weighting.experiments.phylo_signal import run_embedding_tree_distance_analysis
from plm_phylo_weighting.experiments.prepare_data import prepare_data


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, families, check_df, _ = prepare_data(config)
    valid_family_ids = check_df[check_df["ok"]]["FAM"].tolist()
    _, summary_df, size_summary_df = run_embedding_tree_distance_analysis(config, families, valid_family_ids)
    print(summary_df)
    print(size_summary_df.head(30))


if __name__ == "__main__":
    main()
