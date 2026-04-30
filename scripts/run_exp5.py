"""Experiment 5: Feature-label mutual information sweep via center_spread.

Varies the Mldatagen `center_spread` ∈ {0.3, 0.5, 0.7, 1.0} at fixed radius
range and fixed h≈0.4. Smaller center_spread packs label-sphere centers
tighter near the origin → more sphere overlap → lower feature-label MI.

Fixed parameters (paper baseline + Exp 3 conventions):
  N = 3000, |F| = 10, |C| = 20
  label_noise = 0.05  (matches paper's Hyperspheres_10_10_0)
  irrelevant_features = 0  (Exp 5 isolates intrinsic MI, not noise dilution)
  radius_range = Mldatagen default (auto)
  h_target = 0.4, seed = 0

Steps per condition:
  1. Generate dataset  →  data/synthetic/exp5/cs{X}/
  2. sweep_homophily   →  data/synthetic/exp5/cs{X}_h0.4/
  3. run_batch.py across all 4 h0.4 directories (GCN + H2GCN × 3 seeds = 24)

Caveat (per experiments2.md): smaller center_spread also raises l_mean,
so multi-label character is not perfectly held constant. Report l_mean
per condition alongside AP.

Run from repo root:
    python scripts/run_exp5.py
    python scripts/run_exp5.py --only-generate
    python scripts/run_exp5.py --only-train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
CENTER_SPREADS = [0.3, 0.5, 0.7, 1.0]
N = 3000
FEATURE_DIM = 10
NUM_LABELS = 20
LABEL_NOISE = 0.05
IRR_FEATURES = 10
SEED = 0
H_TARGET = 0.4

BASE_DIR = os.path.join("data", "synthetic", "exp5")
RESULTS_FILE = os.path.join("results", "exp5_center_spread.csv")

# Standard b-grid; matches Exp 3. Smaller center_spread compresses Hamming
# distances (more overlap → more shared labels), but h≈0.4 stays reachable.
B_GRID = ["0.04", "0.06", "0.08", "0.12", "0.20", "0.35"]


def _cs_tag(cs: float) -> str:
    return f"cs{cs:g}".replace(".", "p")


def _run(cmd: list[str], label: str) -> None:
    print(f"\n[Exp5] {label}")
    print("  $", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def generate_datasets() -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    for cs in CENTER_SPREADS:
        out = os.path.join(BASE_DIR, _cs_tag(cs))
        if os.path.exists(os.path.join(out, "labels.csv")):
            print(f"[Exp5] Skip generate cs={cs} (already exists)")
            continue
        _run(
            [
                sys.executable, "-m", "generators.generate_hypersphere",
                "--n", str(N),
                "--feature-dim", str(FEATURE_DIM),
                "--num-labels", str(NUM_LABELS),
                "--center-spread", str(cs),
                "--label-noise", str(LABEL_NOISE),
                "--irrelevant-features", str(IRR_FEATURES),
                "--seed", str(SEED),
                "--out", out,
            ],
            f"Generate center_spread={cs}",
        )


def build_graphs() -> None:
    for cs in CENTER_SPREADS:
        src = os.path.join(BASE_DIR, _cs_tag(cs))
        out_prefix = os.path.join(BASE_DIR, _cs_tag(cs))
        h_dir = f"{out_prefix}_h{H_TARGET:g}"
        if os.path.exists(os.path.join(h_dir, "edge_index.npy")):
            print(f"[Exp5] Skip sweep cs={cs} h={H_TARGET:g} (already exists)")
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
            f"Sweep h={H_TARGET:g} for center_spread={cs}",
        )


def run_training() -> None:
    datasets = [
        os.path.join(BASE_DIR, f"{_cs_tag(cs)}_h{H_TARGET:g}")
        for cs in CENTER_SPREADS
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
            "--models", "GCN", "H2GCN",
            "--seeds", "0", "1", "2",
            "--epochs", "300",
            "--patience", "30",
            "--output", RESULTS_FILE,
        ],
        "Training: 4 datasets × 2 models × 3 seeds = 24 runs",
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

    print("\n[Exp5] Done.")
    if not args.only_generate:
        print(f"  Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
