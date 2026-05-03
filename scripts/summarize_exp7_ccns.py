"""Compute CCNS matrix summaries for Exp7 and Exp7b graphs.

Writes one row per graph to results/exp7_ccns_summary.csv. The full CCNS
matrix is C x C; this script stores scalar summaries useful for analysis:
diagonal mean, off-diagonal mean, their contrast, and Frobenius norm.

Run from repo root:
    python scripts/summarize_exp7_ccns.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generators.properties import ccns


GRAPH_SETS = [
    ("edge", Path("data/synthetic/exp7_edge_addition")),
    ("clean", Path("data/synthetic/exp7_matched_homophily")),
]
OUT_FILE = Path("results/exp7_ccns_summary.csv")


def should_include(kind: str, dataset: str) -> bool:
    if kind == "edge":
        return dataset.startswith("add")
    return dataset.startswith("clean")


def summarize_ccns_matrix(matrix: np.ndarray) -> dict:
    num_labels = matrix.shape[0]
    diag_mean = float(np.diag(matrix).mean())
    offdiag_mean = float(
        (matrix.sum() - np.trace(matrix)) / (num_labels * (num_labels - 1))
    )
    return {
        "ccns_diag_mean": diag_mean,
        "ccns_offdiag_mean": offdiag_mean,
        "ccns_contrast": diag_mean - offdiag_mean,
        "ccns_fro_norm": float(np.linalg.norm(matrix)),
    }


def main() -> None:
    rows = []
    for kind, base_dir in GRAPH_SETS:
        if not base_dir.exists():
            continue
        for summary_path in sorted(base_dir.glob("*/graph_summary.json")):
            dataset = summary_path.parent.name
            if not should_include(kind, dataset):
                continue

            with open(summary_path) as f:
                graph_summary = json.load(f)
            labels = pd.read_csv(summary_path.parent / "labels.csv").values
            edge_index = np.load(summary_path.parent / "edge_index.npy")

            matrix = ccns(edge_index, labels)
            rows.append(
                {
                    "kind": kind,
                    "dataset": dataset,
                    "actual_h": graph_summary["actual_h"],
                    "avg_degree": graph_summary["avg_degree"],
                    **summarize_ccns_matrix(matrix),
                }
            )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ["kind", "actual_h"], ascending=[True, False]
    )
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(df)} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
