"""Simple edge-removal corruption strategy.

Randomly removes a fixed number of directed edges from a (N, N, R) adjacency
tensor. Uniform sampling over all non-zero cells — no upper-triangle restriction
because KG edges are directed (not symmetric).
"""
import numpy as np


def apply_kg_edge_removal(
    adj_nxnxr: np.ndarray,
    num_corruptions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Remove `num_corruptions` directed edges uniformly at random.

    Parameters
    ----------
    adj_nxnxr      : np.ndarray, shape (N, N, R), float32
    num_corruptions: int — number of edges to remove
    rng            : np.random.Generator

    Returns
    -------
    np.ndarray — corrupted copy, same shape as input
    """
    corrupted = adj_nxnxr.copy()
    edge_positions = list(zip(*np.where(corrupted > 0.5)))   # list of (i, j, r) tuples
    if not edge_positions:
        return corrupted
    chosen_indices = rng.choice(
        len(edge_positions),
        size=min(num_corruptions, len(edge_positions)),
        replace=False,
    )
    for idx in chosen_indices:
        i, j, r = edge_positions[idx]
        corrupted[i, j, r] = 0.0
    return corrupted
