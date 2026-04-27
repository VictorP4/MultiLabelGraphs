"""
Social Distance Attachment edge sampler (Boguna et al. 2004) as used by
Zhao et al. 2023, Section 4, Eq. 2:

    p_ij = 1 / (1 + [b^-1 * d(y_i, y_j)]^alpha)

where d(.,.) is the normalized Hamming distance between multi-label
vectors. Vectorized over all pairs; replaces the legacy O(N^2) Python
loop in sda_from_hypersphere.py.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def build_edges(
    labels: np.ndarray,
    alpha: float,
    b: float,
    seed: int | None = None,
) -> np.ndarray:
    """Sample an SDA graph from binary multi-label vectors.

    Returns a symmetric edge_index of shape (2, 2E) with no self-loops.
    """
    if b <= 0:
        raise ValueError("b must be > 0")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    n = labels.shape[0]
    dists = squareform(pdist(labels, metric="hamming"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 1.0 / (1.0 + (dists / b) ** alpha)

    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(n, k=1)
    mask = rng.uniform(size=iu.shape) < p[iu, ju]
    src, dst = iu[mask], ju[mask]

    return np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])]
    ).astype(np.int64)


def load_labels(data_dir: str) -> np.ndarray:
    return pd.read_csv(os.path.join(data_dir, "labels.csv")).values


def save_edge_index(data_dir: str, edge_index: np.ndarray) -> None:
    np.save(os.path.join(data_dir, "edge_index.npy"), edge_index)
