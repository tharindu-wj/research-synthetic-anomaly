"""TRIC-style triple corruption strategy for (N, N, R) adjacency tensors.

Implements a subset of the TRIC operations (Senaratne et al., ESWC 2023,
Table 1) adapted for the binary adjacency tensor representation:

    A triple (s, p, o) maps to  adj[s, o, r] = 1
    where r is the channel index for predicate p.

Supported operations
--------------------
remove                  Delete a random existing edge.            (TRIC types 1-4)
swap_subject_same_type  Move edge (i→j, r) to (k→j, r)
                        where type[k] == type[i].                 (TRIC type 1)
swap_object_same_type   Move edge (i→j, r) to (i→k, r)
                        where type[k] == type[j].                 (TRIC type 2)
change_relation         Move edge (i,j,r) to (i,j,r').            (TRIC type 3)
swap_subject_diff_type  Move edge (i→j, r) to (k→j, r)
                        where type[k] != type[i].                 (TRIC type 5)
swap_object_diff_type   Move edge (i→j, r) to (i→k, r)
                        where type[k] != type[j].                 (TRIC type 6)
add_spurious            Add a random non-existing off-diagonal
                        edge (type-agnostic).                     (TRIC types 9-10)

Literal-based TRIC types (11-15) are not applicable — this KG has no
attribute literals in the adjacency tensor.
"""
import numpy as np
from typing import Optional


_ALL_OPS = [
    'remove',
    'swap_subject_same_type',
    'swap_object_same_type',
    'change_relation',
    'swap_subject_diff_type',
    'swap_object_diff_type',
    'add_spurious',
]

_TYPE_OPS = {
    'swap_subject_same_type',
    'swap_object_same_type',
    'swap_subject_diff_type',
    'swap_object_diff_type',
}

_DEFAULT_WEIGHTS_WITH_TYPES = {
    'remove':                  2,
    'swap_subject_same_type':  2,
    'swap_object_same_type':   2,
    'change_relation':         1,
    'swap_subject_diff_type':  1,
    'swap_object_diff_type':   1,
    'add_spurious':            1,
}

_DEFAULT_WEIGHTS_NO_TYPES = {
    'remove':          3,
    'change_relation': 2,
    'add_spurious':    1,
}


def apply_tric_corruption(
    adj_nxnxr: np.ndarray,
    num_corruptions: int,
    rng: np.random.Generator,
    node_types: Optional[dict] = None,
    op_weights: Optional[dict] = None,
) -> np.ndarray:
    """Apply TRIC-style triple corruption to a (N, N, R) adjacency tensor.

    Each corruption step:
      1. Randomly selects one operation from the weighted distribution.
      2. Picks a random existing edge (or non-edge for add_spurious).
      3. Applies the transformation in-place on the working copy.
      4. Skips gracefully if no valid target exists for the chosen op
         (e.g. no same-type swap partner available).

    Parameters
    ----------
    adj_nxnxr      : np.ndarray, shape (N, N, R), float32
    num_corruptions: int — number of corruption steps to attempt
    rng            : np.random.Generator
    node_types     : dict[int, str] or None
                     row_index → entity type string (e.g. 'Person', 'Country').
                     Required for type-aware ops; those ops are silently dropped
                     from the distribution when node_types is None.
    op_weights     : dict[str, float] or None
                     Maps operation name → relative weight. Unrecognised keys are
                     ignored. If None, uses the built-in defaults.

    Returns
    -------
    np.ndarray — corrupted copy, same shape as input
    """
    corrupted = adj_nxnxr.copy()
    N, _, R = corrupted.shape

    # ── Resolve operation distribution ───────────────────────────────────────
    if op_weights is None:
        base_weights = (
            _DEFAULT_WEIGHTS_WITH_TYPES if node_types is not None
            else _DEFAULT_WEIGHTS_NO_TYPES
        )
    else:
        base_weights = {k: v for k, v in op_weights.items() if k in _ALL_OPS}

    # Drop type-aware ops when node_types is unavailable
    if node_types is None:
        active_weights = {k: v for k, v in base_weights.items() if k not in _TYPE_OPS}
    else:
        active_weights = dict(base_weights)

    if not active_weights:
        return corrupted

    ops   = list(active_weights.keys())
    probs = np.array([active_weights[o] for o in ops], dtype=float)
    probs /= probs.sum()

    # ── Helper: current edge and non-edge lists ───────────────────────────────
    def _edges():
        ii, jj, rr = np.where(corrupted > 0.5)
        mask = ii != jj   # exclude diagonal (no self-loops)
        return list(zip(ii[mask], jj[mask], rr[mask]))

    def _non_edges():
        ii, jj, rr = np.where(corrupted < 0.5)
        mask = ii != jj   # exclude diagonal
        return list(zip(ii[mask], jj[mask], rr[mask]))

    # ── Apply corruption steps ────────────────────────────────────────────────
    for _ in range(num_corruptions):
        op = ops[rng.choice(len(ops), p=probs)]

        if op == 'remove':
            edge_list = _edges()
            if not edge_list:
                continue
            i, j, r = edge_list[rng.integers(len(edge_list))]
            corrupted[i, j, r] = 0.0

        elif op == 'add_spurious':
            non_edge_list = _non_edges()
            if not non_edge_list:
                continue
            i, j, r = non_edge_list[rng.integers(len(non_edge_list))]
            corrupted[i, j, r] = 1.0

        elif op == 'change_relation':
            edge_list = _edges()
            if not edge_list:
                continue
            i, j, r = edge_list[rng.integers(len(edge_list))]
            other_relations = [r2 for r2 in range(R) if r2 != r and corrupted[i, j, r2] < 0.5]
            if not other_relations:
                continue
            r2 = other_relations[rng.integers(len(other_relations))]
            corrupted[i, j, r]  = 0.0
            corrupted[i, j, r2] = 1.0

        elif op == 'swap_subject_same_type':
            edge_list = _edges()
            if not edge_list:
                continue
            i, j, r = edge_list[rng.integers(len(edge_list))]
            candidates = [
                k for k in range(N)
                if k != i and k != j
                and node_types.get(k) == node_types.get(i)
                and corrupted[k, j, r] < 0.5
            ]
            if not candidates:
                continue
            k = candidates[rng.integers(len(candidates))]
            corrupted[i, j, r] = 0.0
            corrupted[k, j, r] = 1.0

        elif op == 'swap_object_same_type':
            edge_list = _edges()
            if not edge_list:
                continue
            i, j, r = edge_list[rng.integers(len(edge_list))]
            candidates = [
                k for k in range(N)
                if k != j and k != i
                and node_types.get(k) == node_types.get(j)
                and corrupted[i, k, r] < 0.5
            ]
            if not candidates:
                continue
            k = candidates[rng.integers(len(candidates))]
            corrupted[i, j, r] = 0.0
            corrupted[i, k, r] = 1.0

        elif op == 'swap_subject_diff_type':
            edge_list = _edges()
            if not edge_list:
                continue
            i, j, r = edge_list[rng.integers(len(edge_list))]
            candidates = [
                k for k in range(N)
                if k != i and k != j
                and node_types.get(k) != node_types.get(i)
                and corrupted[k, j, r] < 0.5
            ]
            if not candidates:
                continue
            k = candidates[rng.integers(len(candidates))]
            corrupted[i, j, r] = 0.0
            corrupted[k, j, r] = 1.0

        elif op == 'swap_object_diff_type':
            edge_list = _edges()
            if not edge_list:
                continue
            i, j, r = edge_list[rng.integers(len(edge_list))]
            candidates = [
                k for k in range(N)
                if k != j and k != i
                and node_types.get(k) != node_types.get(j)
                and corrupted[i, k, r] < 0.5
            ]
            if not candidates:
                continue
            k = candidates[rng.integers(len(candidates))]
            corrupted[i, j, r] = 0.0
            corrupted[i, k, r] = 1.0

    return corrupted
