from __future__ import annotations

import gzip
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def download_if_missing(url: str, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        urllib.request.urlretrieve(url, target)

    return target


def parse_pandit_summary(path: str | Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    with gzip.open(path, "rt", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            tag = line[:3]
            value = line[5:].strip() if len(line) > 5 else ""

            if tag == "FAM":
                if current is not None:
                    records.append(current)
                current = {"FAM": value}
            elif current is not None:
                if tag in {"PID", "DES"}:
                    current[tag] = value
                elif tag in {"ANO", "ALN", "DNO", "DLN"}:
                    try:
                        current[tag] = int(value)
                    except ValueError:
                        current[tag] = np.nan
                elif tag == "APH":
                    current["has_APH"] = True
                elif tag == "DPH":
                    current["has_DPH"] = True

        if current is not None:
            records.append(current)

    df = pd.DataFrame(records)
    df["has_APH"] = df["has_APH"].fillna(False)
    df["has_DPH"] = df["has_DPH"].fillna(False)
    return df


def parse_selected_families(path: str | Path, selected_fams: list[str]) -> dict[str, dict[str, Any]]:
    selected = set(selected_fams)
    families: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    current_name: str | None = None

    with gzip.open(path, "rt", errors="replace") as file:
        for raw_line in file:
            line = raw_line.rstrip("\n")
            if not line:
                continue

            tag = line[:3]
            value = line[5:].strip() if len(line) > 5 else ""

            if tag == "FAM":
                if current is not None and current["FAM"] in selected:
                    families[current["FAM"]] = current

                current = {"FAM": value, "LNK": [], "ASQ": {}, "DSQ": {}}
                current_name = None
            elif current is not None:
                if tag in {"PID", "DES", "APH", "ATP", "DPH", "DTP"}:
                    current[tag] = value
                elif tag in {"ANO", "ALN", "DNO", "DLN"}:
                    try:
                        current[tag] = int(value)
                    except ValueError:
                        current[tag] = None
                elif tag == "LNK":
                    current["LNK"].append(value)
                elif tag == "NAM":
                    current_name = value
                elif tag == "ASQ" and current_name is not None:
                    current["ASQ"][current_name] = value
                elif tag == "DSQ" and current_name is not None:
                    current["DSQ"][current_name] = value

        if current is not None and current["FAM"] in selected:
            families[current["FAM"]] = current

    return families


def extract_leaf_names_from_newick(tree_text: str) -> list[str]:
    pattern = r"(?<=[(,])([^():,;]+)(?::[0-9.eE+-]+)?"
    return re.findall(pattern, tree_text)


def check_families(families: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for fam_id, fam in families.items():
        leaves = extract_leaf_names_from_newick(fam["APH"])
        rows.append({
            "FAM": fam_id,
            "PID": fam.get("PID"),
            "ANO": fam.get("ANO"),
            "n_asq": len(fam.get("ASQ", {})),
            "n_lnk": len(fam.get("LNK", [])),
            "n_leaves": len(leaves),
            "unique_leaves": len(set(leaves)),
            "has_APH": "APH" in fam,
        })

    df = pd.DataFrame(rows)
    df["ok"] = (
        (df["ANO"] == df["n_asq"])
        & (df["ANO"] == df["n_leaves"])
        & (df["n_leaves"] == df["unique_leaves"])
        & df["has_APH"]
    )
    return df
