"""Experiment 3: Feature and label space dimensions sweep.

2D grid: |F| ∈ {10, 50, 200} × |C| ∈ {5, 20, 100} at fixed h≈0.4.
9 conditions × 2 models × 3 seeds = 54 training runs.

Design choices:
  label_noise = 0.05  (matches paper's Hyperspheres_10_10_0 dataset)
  irrelevant_features = 0  (Mirr confounds the |F| axis: fixed Mirr=10 would
    give 50%/83%/95% relevant fractions at |F|=10/50/200, biasing the sweep;
    Exp 5 tests the MI/noise axis separately)
  N = 3000, h_target = 0.4, seed = 0

Steps per condition:
  1. Generate dataset  →  data/synthetic/exp3/f{F}_c{C}/
  2. sweep_homophily   →  data/synthetic/exp3/f{F}_c{C}_h0.4/
  3. run_batch.py across all 9 h0.4 directories

Run from repo root:
    python scripts/run_exp3.py
    python scripts/run_exp3.py --only-generate
    python scripts/run_exp3.py --only-train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
FEATURE_DIMS = [10, 50, 200]
NUM_LABELS = [5, 20, 100]
N = 3000
LABEL_NOISE = 0.05
SEED = 0
H_TARGET = 0.4

BASE_DIR = os.path.join("data", "synthetic", "exp3")
RESULTS_FILE = os.path.join("results", "exp3_feature_label_dims.csv")

# Wider b-grid than the default to handle different |C| regimes:
#   |C|=5  → coarse Hamming distances → may need larger b
#   |C|=100 → fine Hamming distances → standard b range works
B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35", "0.55"]


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp3] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def generate_datasets() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    for f in FEATURE_DIMS:
        for c in NUM_LABELS:
            out = os.path.join(BASE_DIR, f"f{f}_c{c}")
            if os.path.exists(os.path.join(out, "labels.csv")):
                print(f"[Exp3] Skip generate f={f} c={c} (already exists)")
                continue
            _run(
                [
                    sys.executable, "-m", "generators.generate_hypersphere",
                    "--n", str(N),
                    "--feature-dim", str(f),
                    "--num-labels", str(c),
                    "--label-noise", str(LABEL_NOISE),
                    "--irrelevant-features", "0",
                    "--seed", str(SEED),
                    "--out", out,
                ],
                f"Generate |F|={f} |C|={c}",
            )


def build_graphs() -> None:
    for f in FEATURE_DIMS:
        for c in NUM_LABELS:
            src = os.path.join(BASE_DIR, f"f{f}_c{c}")
            out_prefix = os.path.join(BASE_DIR, f"f{f}_c{c}")
            h_dir = f"{out_prefix}_h{H_TARGET:g}"
            if os.path.exists(os.path.join(h_dir, "edge_index.npy")):
                print(f"[Exp3] Skip sweep f={f} c={c} h={H_TARGET:g} (already exists)")
                continue
            _run(
                [
                    sys.executable, "-m", "generators.sweep_homophily",
                    "--data", src,
                    "--out-prefix", out_prefix,
                    "--targets", str(H_TARGET),
                    "--b-grid", *B_GRID,
                    "--n-trials", "5",
                    "--seed", str(SEED),
                ],
                f"Sweep h={H_TARGET:g} for |F|={f} |C|={c}",
            )


def run_training() -> None:
    datasets = [
        os.path.join(BASE_DIR, f"f{f}_c{c}_h{H_TARGET:g}")
        for f in FEATURE_DIMS
        for c in NUM_LABELS
    ]
    missing = [d for d in datasets if not os.path.exists(d)]
    if missing:
        print("ERROR: missing graph directories (run without --only-train first):",
              file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)

    _run(
        [
            sys.executable, "run_batch.py",
            "--datasets", *datasets,
            "--models", "H2GCN",
            "--seeds", "0", "1", "2",
            "--epochs", "300",
            "--patience", "30",
            "--output", RESULTS_FILE,
        ],
        "Training: 9 datasets × 2 models × 3 seeds = 54 runs",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--only-generate", action="store_true",
                   help="only generate datasets + graphs, skip training")
    g.add_argument("--only-train", action="store_true",
                   help="only train (assume datasets+graphs already exist)")
    args = p.parse_args(argv)

    if not args.only_train:
        generate_datasets()
        build_graphs()

    if not args.only_generate:
        run_training()

    print("\n[Exp3] Done.")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
