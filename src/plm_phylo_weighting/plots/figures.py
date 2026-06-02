from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_methods(summary_df: pd.DataFrame, out_path: str | Path, top_n: int = 20) -> None:
    if len(summary_df) == 0:
        return

    top_df = summary_df.head(top_n).copy()
    top_df["label"] = (
        top_df["model_key"].astype(str)
        + " | "
        + top_df["phylo_method"].astype(str)
        + " | "
        + top_df["method_name"].astype(str)
    )

    if "feature_group" in top_df.columns:
        top_df["label"] = top_df["label"] + " | " + top_df["feature_group"].astype(str)

    plt.figure(figsize=(10, 8))
    plt.barh(top_df["label"][::-1], top_df["mean_spearman"][::-1])
    plt.xlabel("Mean Spearman correlation")
    plt.title("Top methods")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_model_comparison(summary_df: pd.DataFrame, out_path: str | Path) -> None:
    if len(summary_df) == 0:
        return

    plot_df = (
        summary_df
        .groupby(["model_key", "method_group"])
        ["mean_spearman"]
        .max()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for method_group, part in plot_df.groupby("method_group"):
        plt.plot(part["model_key"], part["mean_spearman"], marker="o", label=method_group)

    plt.xlabel("PLM model")
    plt.ylabel("Best mean Spearman")
    plt.title("Best score by model and method group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_size_effect(size_analysis_df: pd.DataFrame, out_path: str | Path, top_n: int = 30) -> None:
    if len(size_analysis_df) == 0:
        return

    plot_df = size_analysis_df.sort_values("mean_spearman", ascending=False).head(top_n).copy()
    plot_df["label"] = (
        plot_df["source"].astype(str)
        + " | "
        + plot_df["model_key"].astype(str)
        + " | "
        + plot_df["phylo_method"].astype(str)
        + " | "
        + plot_df["method_name"].astype(str)
        + " | "
        + plot_df["size_bin"].astype(str)
    )

    plt.figure(figsize=(12, 10))
    plt.barh(plot_df["label"][::-1], plot_df["mean_spearman"][::-1])
    plt.xlabel("Mean Spearman correlation")
    plt.title("Best method performance by family size bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_embedding_tree_by_size(signal_size_df: pd.DataFrame, out_path: str | Path) -> None:
    if len(signal_size_df) == 0:
        return

    plt.figure(figsize=(10, 6))

    for model_key, part in signal_size_df.groupby("model_key"):
        labels = part["size_bin"].astype(str)
        plt.plot(labels, part["mean_spearman"], marker="o", label=model_key)

    plt.xlabel("Family size bin")
    plt.ylabel("Mean Spearman correlation")
    plt.title("Embedding distance vs tree distance by family size")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_local_feature_signal(feature_summary_df: pd.DataFrame, out_path: str | Path, top_n: int = 30) -> None:
    if len(feature_summary_df) == 0:
        return

    plot_df = feature_summary_df.head(top_n).copy()
    plot_df["label"] = (
        plot_df["model_key"].astype(str)
        + " | "
        + plot_df["phylo_method"].astype(str)
        + " | "
        + plot_df["feature"].astype(str)
    )

    plt.figure(figsize=(12, 10))
    plt.barh(plot_df["label"][::-1], plot_df["mean_spearman"][::-1])
    plt.xlabel("Mean Spearman correlation")
    plt.title("Local PLM features vs phylogenetic weights")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
