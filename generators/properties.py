"""
Graph and label property computations for multi-label datasets.

Implements:
- Multi-label label homophily (Zhao et al. 2023, Def. 1)
- Cross-Class Neighborhood Similarity (Zhao et al. 2023, Def. 2)
- Paper Table-1-style dataset summary
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def _undirected_pairs(edge_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique (src, dst) pairs with src < dst."""
    src, dst = edge_index
    mask = src < dst
    return src[mask], dst[mask]


def _symmetric_adj(edge_index: np.ndarray, num_nodes: int) -> csr_matrix:
    src, dst = edge_index
    both_src = np.concatenate([src, dst])
    both_dst = np.concatenate([dst, src])
    data = np.ones(both_src.shape[0], dtype=np.float64)
    adj = csr_matrix((data, (both_src, both_dst)), shape=(num_nodes, num_nodes))
    adj.data = np.minimum(adj.data, 1.0)
    return adj


def label_homophily(edge_index: np.ndarray, labels: np.ndarray) -> float:
    """Average Jaccard similarity of label sets over undirected edges."""
    src, dst = _undirected_pairs(edge_index)
    if len(src) == 0:
        return float("nan")
    ls = labels[src].astype(bool)
    ld = labels[dst].astype(bool)
    inter = np.logical_and(ls, ld).sum(axis=1)
    union = np.logical_or(ls, ld).sum(axis=1)
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else 0.0


def ccns(edge_index: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cross-Class Neighborhood Similarity matrix, shape (C, C)."""
    n, c = labels.shape
    adj = _symmetric_adj(edge_index, n)
    neigh = adj @ labels.astype(np.float64)

    norms = np.linalg.norm(neigh, axis=1, keepdims=True)
    neigh_unit = np.divide(neigh, norms, out=np.zeros_like(neigh), where=norms > 0)

    l_count = labels.sum(axis=1).astype(np.float64)
    inv_l = np.divide(1.0, l_count, out=np.zeros_like(l_count), where=l_count > 0)
    alpha = labels.astype(np.float64) * inv_l[:, None]

    cos_sim = neigh_unit @ neigh_unit.T
    num = alpha.T @ cos_sim @ alpha - alpha.T @ alpha

    v_size = labels.sum(axis=0).astype(np.float64)
    denom = np.outer(v_size, v_size)
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


def clustering_coefficient(edge_index: np.ndarray, num_nodes: int) -> float:
    import networkx as nx

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    src, dst = _undirected_pairs(edge_index)
    g.add_edges_from(zip(src.tolist(), dst.tolist()))
    return float(nx.average_clustering(g))


def summarize(
    edge_index: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray | None = None,
    include_clustering: bool = False,
) -> dict:
    """Paper Table-1-style dataset statistics."""
    n, c = labels.shape
    src, dst = _undirected_pairs(edge_index)
    num_edges = int(len(src))
    counts = labels.sum(axis=1)

    stats = {
        "num_nodes": int(n),
        "num_edges": num_edges,
        "avg_degree": 2.0 * num_edges / n if n > 0 else 0.0,
        "density": 2.0 * num_edges / (n * (n - 1)) if n > 1 else 0.0,
        "num_labels": int(c),
        "l_mean": float(counts.mean()),
        "l_min": int(counts.min()),
        "l_max": int(counts.max()),
        "unlabeled_fraction": float((counts == 0).mean()),
        "label_homophily": label_homophily(edge_index, labels),
    }
    if features is not None:
        stats["feature_dim"] = int(features.shape[1])
        stats["feature_sparsity"] = float((features == 0).mean())
    if include_clustering:
        stats["clustering_coefficient"] = clustering_coefficient(edge_index, n)
    return stats
