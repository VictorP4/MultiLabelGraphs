"""Create figures for Experiment 7.

Reads Exp7 result/summary CSVs and writes plots to figures/.

Run from repo root:
    python scripts/plot_exp7_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


EDGE_RESULTS = Path("results/exp7_edge_addition.csv")
CLEAN_RESULTS = Path("results/exp7_matched_homophily.csv")
CCNS_RESULTS = Path("results/exp7_ccns_summary.csv")
OUT_DIR = Path("figures")

EDGE_RATE = {
    "add0_h0.6": 0,
    "add0p1_h0.6": 10,
    "add0p25_h0.6": 25,
    "add0p5_h0.6": 50,
    "add1_h0.6": 100,
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def _load_edge_results() -> pd.DataFrame:
    df = pd.read_csv(EDGE_RESULTS)
    df["edge_added_pct"] = df["dataset"].map(EDGE_RATE)
    agg = (
        df.groupby(["dataset", "model", "edge_added_pct"])
        .agg(
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            macro_mean=("macro_f1", "mean"),
            macro_std=("macro_f1", "std"),
        )
        .reset_index()
        .sort_values(["edge_added_pct", "model"])
    )
    return agg


def _load_clean_results() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_RESULTS)
    agg = (
        df.groupby(["dataset", "model"])
        .agg(ap_mean=("ap", "mean"), ap_std=("ap", "std"))
        .reset_index()
    )
    return agg


def _load_ccns() -> pd.DataFrame:
    df = pd.read_csv(CCNS_RESULTS)
    df["edge_added_pct"] = df["dataset"].map(EDGE_RATE)
    return df


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Wrote {path}")


def plot_edge_vs_clean(edge: pd.DataFrame, clean: pd.DataFrame, ccns: pd.DataFrame) -> None:
    graph_props = ccns[ccns["kind"].isin(["edge", "clean"])][
        ["kind", "dataset", "actual_h", "avg_degree", "ccns_contrast"]
    ]
    edge_plot = edge[["dataset", "model", "ap_mean", "ap_std"]].copy()
    edge_plot["kind"] = "edge"
    clean_plot = clean[["dataset", "model", "ap_mean", "ap_std"]].copy()
    clean_plot["kind"] = "clean"
    plot_df = pd.concat([edge_plot, clean_plot], ignore_index=True).merge(
        graph_props, on=["kind", "dataset"], how="left"
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, model in zip(axes, ["GCN", "H2GCN"]):
        for kind, label, marker in [
            ("edge", "Edge-added", "o"),
            ("clean", "Clean SDA", "s"),
        ]:
            part = plot_df[(plot_df["model"] == model) & (plot_df["kind"] == kind)]
            part = part.sort_values("actual_h")
            ax.errorbar(
                part["actual_h"],
                part["ap_mean"],
                yerr=part["ap_std"],
                marker=marker,
                capsize=3,
                linewidth=2,
                label=label,
            )
        ax.set_title(model)
        ax.set_xlabel("Measured homophily")
        ax.invert_xaxis()
    axes[0].set_ylabel("Macro average precision")
    axes[1].legend(title="Graph type")
    _save(fig, "exp7_edge_vs_clean_matched_homophily.png")


def plot_homophily_vs_ccns(ccns: pd.DataFrame) -> None:
    plot_df = ccns[ccns["kind"].isin(["edge", "clean"])].copy()
    labels = {"edge": "Edge-added", "clean": "Clean SDA"}
    markers = {"edge": "o", "clean": "s"}

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for kind, part in plot_df.groupby("kind"):
        part = part.sort_values("actual_h")
        ax.plot(
            part["actual_h"],
            part["ccns_contrast"],
            marker=markers[kind],
            linewidth=2,
            label=labels[kind],
        )
    ax.set_xlabel("Measured homophily")
    ax.set_ylabel("CCNS contrast")
    ax.invert_xaxis()
    ax.legend(title="Graph type")
    _save(fig, "exp7_homophily_vs_ccns_contrast.png")


def main() -> None:
    _style()
    edge = _load_edge_results()
    clean = _load_clean_results()
    ccns = _load_ccns()

    plot_edge_vs_clean(edge, clean, ccns)
    plot_homophily_vs_ccns(ccns)


if __name__ == "__main__":
    main()
