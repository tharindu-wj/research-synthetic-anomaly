"""Build (clean_triple, target_triple) training pairs directly from a triple list.

Replaces the old `build_kg_paired_dataset` + `build_triple_pair_tensor` flow,
which went through an `(N, N, R)` adjacency intermediate. This version is
triple-level end-to-end - the only representation that scales to FB15k-237.

Per round:
  1. Optionally permute entity indices (data augmentation across many KGs).
  2. Apply `corruption_fn` to the (permuted) triple list - returns
     (corrupted_triples, edit_records).
  3. Harvest the substitution edits (records where both clean and corrupted
     triples are populated) as (clean, corrupted) training pairs.
"""
from typing import Callable, List, Tuple

import numpy as np
import torch


Triple = Tuple[int, int, int]


def build_triple_pair_dataset(
    triples: List[Triple],
    num_pairs: int,
    num_corruptions: int,
    rng: np.random.Generator,
    *,
    num_entities: int,
    num_relations: int,
    corruption_fn: Callable,
    permute_entities: bool = True,
) -> torch.Tensor:
    """Build a (M, 6) long tensor of (clean, corrupted) triple index pairs.

    Parameters
    ----------
    triples         : list of clean (h_idx, r_idx, t_idx) tuples
    num_pairs       : int  number of (permutation, corruption) rounds
    num_corruptions : int  passed to corruption_fn
    rng             : np.random.Generator
    num_entities    : |E|
    num_relations   : |R|
    corruption_fn   : callable(triples, num_corruptions, rng) ->
                      (corrupted_triples, edit_records).
                      Pre-bind kwargs via functools.partial.
    permute_entities: bool. If True, each round randomly remaps entity
                      indices before corrupting - data augmentation so the
                      model sees the same KG structure under many entity ids.

    Returns
    -------
    torch.Tensor of shape (M, 6), dtype long.
    Columns: [c_h, c_r, c_t, t_h, t_r, t_t].
    M depends on how many substitution-style edits succeeded across all
    rounds (pure adds/removes are excluded - they have no (clean, corrupted)
    correspondence).
    """
    all_rows: List[List[int]] = []

    for _ in range(num_pairs):
        if permute_entities:
            perm = rng.permutation(num_entities)
            cur_triples = [(int(perm[h]), r, int(perm[t])) for h, r, t in triples]
        else:
            cur_triples = list(triples)

        _corrupted, edits = corruption_fn(cur_triples, num_corruptions, rng)

        for clean_tri, corr_tri in edits:
            if clean_tri is None or corr_tri is None:
                continue   # pure add/remove - no (clean, corrupted) pair
            all_rows.append([
                clean_tri[0], clean_tri[1], clean_tri[2],
                corr_tri[0],  corr_tri[1],  corr_tri[2],
            ])

    if not all_rows:
        return torch.zeros((0, 6), dtype=torch.long)
    return torch.tensor(all_rows, dtype=torch.long)
