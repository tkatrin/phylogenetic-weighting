from __future__ import annotations

import argparse

from plm_phylo_weighting.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    return parser.parse_args()

from plm_phylo_weighting.experiments.prepare_data import prepare_data


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    selected_df, families, check_df, phylo_weights_df = prepare_data(config)

    print("selected:", len(selected_df))
    print("valid:", int(check_df["ok"].sum()))
    print("phylo weights:", len(phylo_weights_df))


if __name__ == "__main__":
    main()
