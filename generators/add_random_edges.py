"""Add uniformly random edges to an existing synthetic graph directory.

Copies features.csv and labels.csv from --data to --out, writes edge_index.npy,
and writes graph_summary.json using the standard property summary shape.

Example:
    python -m generators.add_random_edges \
        --data data/synthetic/exp7_edge_addition/base_h0.6 \
        --out data/synthetic/exp7_edge_addition/add0p25_h0.6 \
        --fraction 0.25 \
        --seed 2500
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from .properties import summarize


def _undirected_pairs(edge_index: np.ndarray) -> set[tuple[int, int]]:
    src, dst = edge_index
    pairs: set[tuple[int, int]] = set()
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    return pairs


def _edge_index_from_pairs(pairs: set[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.array(sorted(pairs), dtype=np.int64)
    src = np.concatenate([arr[:, 0], arr[:, 1]])
    dst = np.concatenate([arr[:, 1], arr[:, 0]])
    return np.stack([src, dst]).astype(np.int64)


def add_random_edges(
    edge_index: np.ndarray,
    num_nodes: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    if fraction < 0:
        raise ValueError("fraction must be >= 0")

    pairs = _undirected_pairs(edge_index)
    original_edges = len(pairs)
    target_add = int(round(original_edges * fraction))
    if target_add == 0:
        return _edge_index_from_pairs(pairs)

    max_edges = num_nodes * (num_nodes - 1) // 2
    if original_edges + target_add > max_edges:
        raise ValueError("requested more edges than a simple graph can hold")

    rng = np.random.default_rng(seed)
    added = 0
    while added < target_add:
        remaining = target_add - added
        batch = max(remaining * 3, 1024)
        u = rng.integers(0, num_nodes, size=batch)
        v = rng.integers(0, num_nodes, size=batch)
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b:
                continue
            x, y = (a, b) if a < b else (b, a)
            if (x, y) in pairs:
                continue
            pairs.add((x, y))
            added += 1
            if added == target_add:
                break

    return _edge_index_from_pairs(pairs)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add uniformly random edges to an existing graph directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", required=True,
                   help="source graph directory containing labels.csv and edge_index.npy")
    p.add_argument("--out", required=True,
                   help="output graph directory")
    p.add_argument("--fraction", type=float, required=True,
                   help="fraction of original undirected edge count to add")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    for fname in ("features.csv", "labels.csv"):
        src = os.path.join(args.data, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, fname))

    labels = pd.read_csv(os.path.join(args.data, "labels.csv")).values
    edge_path = os.path.join(args.out, "edge_index.npy")
    if os.path.exists(edge_path):
        edge_index = np.load(edge_path)
        print(f"Reusing existing edge_index.npy in {args.out}")
    else:
        base_edge_index = np.load(os.path.join(args.data, "edge_index.npy"))
        edge_index = add_random_edges(
            base_edge_index,
            num_nodes=labels.shape[0],
            fraction=args.fraction,
            seed=args.seed,
        )
        np.save(edge_path, edge_index)

    stats = summarize(edge_index, labels)
    with open(os.path.join(args.data, "graph_summary.json")) as f:
        base_summary = json.load(f)
    stats.update({
        "alpha": base_summary["alpha"],
        "b": base_summary["b"],
        "target_h": base_summary["target_h"],
        "actual_h": stats["label_homophily"],
    })

    with open(os.path.join(args.out, "graph_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote edge-added graph to {args.out}")
    print(f"  fraction: {args.fraction:g}")
    print(f"  num_edges: {stats['num_edges']}")
    print(f"  label_homophily: {stats['label_homophily']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
