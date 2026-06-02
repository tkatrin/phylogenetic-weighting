from __future__ import annotations

import argparse

from plm_phylo_weighting.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()

from plm_phylo_weighting.experiments.compute_embeddings import compute_all_embeddings
from plm_phylo_weighting.experiments.prepare_data import prepare_data


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, families, check_df, _ = prepare_data(config)
    valid_family_ids = check_df[check_df["ok"]]["FAM"].tolist()
    compute_all_embeddings(config, families, valid_family_ids)


if __name__ == "__main__":
    main()
