"""Detect whether Mldatagen label noise (mu parameter) was applied to the paper's data.

Reconstructs per-label spheres from data (centroid + max-radius), re-derives geometric
labels, and measures disagreement with the actual labels.csv.

Expected results:
  ~0%   disagreement -> labels are clean
  ~5%   disagreement -> mu=0.05 was used
  ~10%  disagreement -> mu=0.10 was used

Usage:
    python scripts/check_label_noise.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from generators.hypersphere import generate

MREL = 10  # Hyperspheres_10_10_0: 10 relevant + 10 irrelevant features


def disagreement(features_rel: np.ndarray, labels: np.ndarray) -> dict:
    """Calibrated-radius reconstruction: binary-search the radius per label so
    exactly n_pos nodes are selected, then compare to actual labels."""
    n, c = labels.shape
    geom_labels = np.zeros_like(labels)
    for k in range(c):
        members = features_rel[labels[:, k] == 1]
        n_pos = len(members)
        if n_pos == 0:
            continue
        center = members.mean(axis=0)
        dists_all = cdist(features_rel, center[None, :]).ravel()
        r_lo, r_hi = dists_all.min(), dists_all.max()
        for _ in range(40):
            r_mid = (r_lo + r_hi) / 2.0
            if (dists_all <= r_mid).sum() < n_pos:
                r_lo = r_mid
            else:
                r_hi = r_mid
        geom_labels[:, k] = (dists_all <= r_hi).astype(np.int8)
    flips = (labels != geom_labels)
    return {
        "overall_flip_rate":         float(flips.mean()),
        "per_label_flip_rate_mean":  float(flips.mean(axis=0).mean()),
        "frac_nodes_with_any_flip":  float(flips.any(axis=1).mean()),
        "l_mean_observed":           float(labels.sum(axis=1).mean()),
        "l_mean_geometric":          float(geom_labels.sum(axis=1).mean()),
    }


def report(name: str, features: np.ndarray, labels: np.ndarray, mrel: int) -> None:
    print(f"\n=== {name} ===")
    stats = disagreement(features[:, :mrel], labels)
    for key, val in stats.items():
        print(f"  {key}: {val:.4f}")


# Paper's data
paper_features = pd.read_csv("data/Hyperspheres_10_10_0/features.csv").values
paper_labels   = pd.read_csv("data/Hyperspheres_10_10_0/labels.csv").values.astype(np.int8)
report("Paper Hyperspheres_10_10_0", paper_features, paper_labels, MREL)

# Clean baseline (matching paper params: n=3000, 10 relevant, 20 labels, 10 irrelevant)
ours = generate(n=3000, feature_dim=10, num_labels=20, irrelevant_features=10, seed=0)
report("Our generator (seed=0, clean)", ours.features, ours.labels, MREL)

# Noisy baseline with mu=0.05 (Mldatagen default) to validate test sensitivity
ours_noisy = generate(n=3000, feature_dim=10, num_labels=20, irrelevant_features=10,
                      label_noise=0.05, seed=0)
report("Our generator (seed=0, mu=0.05)", ours_noisy.features, ours_noisy.labels, MREL)

print()
print("Interpretation (calibrated-radius method; l_mean matches by construction):")
print("  Baseline reconstruction noise ~4-5% (our clean generator)")
print("  Paper flip_rate - baseline ~= actual label noise rate")
print("  excess ~5%  -> mu=0.05 likely applied to paper data")
print("  excess ~10% -> mu=0.10 likely applied to paper data")
