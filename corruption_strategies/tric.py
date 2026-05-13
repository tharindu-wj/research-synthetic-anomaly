"""TRIC-style triple corruption operating on a triple set (no adjacency tensor).

Implements a subset of the TRIC operations (Senaratne et al., ESWC 2023, Table 1):

    remove                  Delete a random existing triple.        (TRIC 1-4)
    swap_subject_same_type  (h, r, t) -> (h', r, t),
                            type[h'] == type[h].                    (TRIC 1)
    swap_object_same_type   (h, r, t) -> (h, r, t'),
                            type[t'] == type[t].                    (TRIC 2)
    change_relation         (h, r, t) -> (h, r', t).                (TRIC 3)
    swap_subject_diff_type  (h, r, t) -> (h', r, t),
                            type[h'] != type[h].                    (TRIC 5)
    swap_object_diff_type   (h, r, t) -> (h, r, t'),
                            type[t'] != type[t].                    (TRIC 6)
    add_spurious            Add a random non-existing triple.       (TRIC 9-10)

Returns BOTH:
  - the modified triple list (corrupted KG)
  - a list of edit records: (clean_triple_or_None, corrupted_triple_or_None)
    documenting each successful edit. Used by the dataset builder to construct
    (clean, corrupted) training pairs without re-diffing.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np


Triple = Tuple[int, int, int]                          # (h_idx, r_idx, t_idx)
EditRecord = Tuple[Optional[Triple], Optional[Triple]]  # (clean, corrupted)


_ALL_OPS = [
    "remove",
    "swap_subject_same_type",
    "swap_object_same_type",
    "change_relation",
    "swap_subject_diff_type",
    "swap_object_diff_type",
    "add_spurious",
]

_TYPE_OPS = {
    "swap_subject_same_type",
    "swap_object_same_type",
    "swap_subject_diff_type",
    "swap_object_diff_type",
}

_DEFAULT_WEIGHTS_WITH_TYPES = {
    "remove":                  2,
    "swap_subject_same_type":  2,
    "swap_object_same_type":   2,
    "change_relation":         1,
    "swap_subject_diff_type":  1,
    "swap_object_diff_type":   1,
    "add_spurious":            1,
}

_DEFAULT_WEIGHTS_NO_TYPES = {
    "remove":          3,
    "change_relation": 2,
    "add_spurious":    1,
}


def apply_tric_corruption(
    triples: List[Triple],
    num_corruptions: int,
    rng: np.random.Generator,
    *,
    num_entities: int,
    num_relations: int,
    node_types: Optional[Dict[int, str]] = None,
    op_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Triple], List[EditRecord]]:
    """Apply TRIC-style corruption to a triple list.

    Parameters
    ----------
    triples         : list of (h_idx, r_idx, t_idx) tuples
    num_corruptions : int  number of corruption steps to attempt
    rng             : np.random.Generator
    num_entities    : |E|  size of entity vocabulary
    num_relations   : |R|  size of relation vocabulary
    node_types      : optional dict[int, str]  entity_idx -> type
                      Required for type-aware ops; those ops are silently
                      dropped if node_types is None.
    op_weights      : optional dict[op_name, float]  relative weights.
                      Defaults depend on whether node_types is provided.

    Returns
    -------
    corrupted_triples : list of (h, r, t) tuples (the modified KG)
    edit_records      : list of (clean, corrupted) edit records.
                        For 'remove', corrupted is None.
                        For 'add_spurious', clean is None.
                        For swaps and change_relation, both are populated.
    """
    if op_weights is None:
        base_weights = (
            _DEFAULT_WEIGHTS_WITH_TYPES if node_types is not None
            else _DEFAULT_WEIGHTS_NO_TYPES
        )
    else:
        base_weights = {k: v for k, v in op_weights.items() if k in _ALL_OPS}

    if node_types is None:
        active_weights = {k: v for k, v in base_weights.items() if k not in _TYPE_OPS}
    else:
        active_weights = dict(base_weights)

    if not active_weights:
        return list(triples), []

    ops   = list(active_weights.keys())
    probs = np.array([active_weights[o] for o in ops], dtype=float)
    probs /= probs.sum()

    # Working copy + fast existence-check set
    result: List[Triple] = list(triples)
    triple_set: set = set(result)
    edits: List[EditRecord] = []

    def _pick_existing_idx() -> Optional[int]:
        if not result:
            return None
        return int(rng.integers(len(result)))

    for _ in range(num_corruptions):
        op = ops[rng.choice(len(ops), p=probs)]

        # ---- Pure delete ----
        if op == "remove":
            idx = _pick_existing_idx()
            if idx is None:
                continue
            removed = result.pop(idx)
            triple_set.discard(removed)
            edits.append((removed, None))
            continue

        # ---- Pure add ----
        if op == "add_spurious":
            for _try in range(50):
                h = int(rng.integers(num_entities))
                t = int(rng.integers(num_entities))
                r = int(rng.integers(num_relations))
                if h != t and (h, r, t) not in triple_set:
                    new_tri = (h, r, t)
                    result.append(new_tri)
                    triple_set.add(new_tri)
                    edits.append((None, new_tri))
                    break
            continue

        # ---- Substitution ops (need an existing triple) ----
        idx = _pick_existing_idx()
        if idx is None:
            continue
        h, r, t = result[idx]
        new_tri: Optional[Triple] = None

        if op == "change_relation":
            cands = [r2 for r2 in range(num_relations)
                     if r2 != r and (h, r2, t) not in triple_set]
            if cands:
                r2 = int(cands[rng.integers(len(cands))])
                new_tri = (h, r2, t)

        elif op == "swap_subject_same_type":
            cands = [k for k in range(num_entities)
                     if k != h and k != t
                     and node_types.get(k) == node_types.get(h)
                     and (k, r, t) not in triple_set]
            if cands:
                k = int(cands[rng.integers(len(cands))])
                new_tri = (k, r, t)

        elif op == "swap_object_same_type":
            cands = [k for k in range(num_entities)
                     if k != t and k != h
                     and node_types.get(k) == node_types.get(t)
                     and (h, r, k) not in triple_set]
            if cands:
                k = int(cands[rng.integers(len(cands))])
                new_tri = (h, r, k)

        elif op == "swap_subject_diff_type":
            cands = [k for k in range(num_entities)
                     if k != h and k != t
                     and node_types.get(k) != node_types.get(h)
                     and (k, r, t) not in triple_set]
            if cands:
                k = int(cands[rng.integers(len(cands))])
                new_tri = (k, r, t)

        elif op == "swap_object_diff_type":
            cands = [k for k in range(num_entities)
                     if k != t and k != h
                     and node_types.get(k) != node_types.get(t)
                     and (h, r, k) not in triple_set]
            if cands:
                k = int(cands[rng.integers(len(cands))])
                new_tri = (h, r, k)

        if new_tri is None:
            continue   # No valid target -> skip this step

        clean_tri = result[idx]
        triple_set.discard(clean_tri)
        triple_set.add(new_tri)
        result[idx] = new_tri
        edits.append((clean_tri, new_tri))

    return result, edits
