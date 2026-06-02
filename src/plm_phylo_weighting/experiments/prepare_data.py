from __future__ import annotations

import pandas as pd

from plm_phylo_weighting.data.pandit import (
    check_families,
    download_if_missing,
    parse_pandit_summary,
    parse_selected_families,
)
from plm_phylo_weighting.data.sampling import select_stratified_families, select_target_mean_families
from plm_phylo_weighting.weights.phylogenetic import PHYLO_METHODS
from plm_phylo_weighting.utils.paths import ensure_dir


def prepare_data(config: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    data_path = download_if_missing(config["data_url"], config["data_path"])
    summary_df = parse_pandit_summary(data_path)

    if "target_mean_ano" in config and config["target_mean_ano"] is not None:
        selected_df = select_target_mean_families(
            summary_df=summary_df,
            n_families=config["n_families"],
            min_ano=config["min_ano"],
            target_mean_ano=config["target_mean_ano"],
            random_state=config["random_state"],
        )
    else:
        selected_df = select_stratified_families(
            summary_df=summary_df,
            n_families=config["n_families"],
            min_ano=config["min_ano"],
            random_state=config["random_state"],
        )

    family_ids = selected_df["FAM"].tolist()
    families = parse_selected_families(data_path, family_ids)
    check_df = check_families(families)
    valid_family_ids = check_df[check_df["ok"]]["FAM"].tolist()

    rows = []

    for fam_id in valid_family_ids:
        fam = families[fam_id]

        for method_name in config["phylo_methods"]:
            method = PHYLO_METHODS[method_name]
            weights = method(fam["APH"], normalize_sum=fam["ANO"])

            for sequence, weight in weights.items():
                rows.append({
                    "FAM": fam_id,
                    "PID": fam.get("PID"),
                    "ANO": fam.get("ANO"),
                    "sequence": sequence,
                    "phylo_method": method_name,
                    "phylo_weight": weight,
                })

    phylo_weights_df = pd.DataFrame(rows)

    work_dir = ensure_dir(config["work_dir"])
    selected_df.to_csv(work_dir / "selected_families.csv", index=False)
    check_df.to_csv(work_dir / "family_checks.csv", index=False)
    phylo_weights_df.to_csv(work_dir / "phylo_weights.csv", index=False)

    return selected_df, families, check_df, phylo_weights_df
