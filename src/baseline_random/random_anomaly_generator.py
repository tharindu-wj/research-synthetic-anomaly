"""Las-Vegas-style random anomaly generator.

Mirrors what ADKGD does internally to make negative training examples
(`generate_anomalous_triples_2` in `baseline/source/dataset.py`): sample
each component of a candidate `(h, r, t)` uniformly from the vocabulary
and retry if the triple already exists in the real graph or is a
self-loop.

Used here for two purposes:
  * Phase-1 deliverable: a stub TSV the user can ship to the ADKGD repo
    *before* the GAN exists, to rehearse the cluster integration
    end-to-end (proves the file format, the loader hook, and the
    `--neg_source gan` plumbing all work).
  * Honest control in the metric comparison: ADKGD's existing random
    negatives come from this same scheme, so if the GAN's TSV outperforms
    a fresh random TSV produced here, we know the win comes from
    structure rather than from a re-run of ADKGD on a different seed.
"""
from __future__ import annotations

from typing import List, Set, Tuple

import numpy as np


Triple = Tuple[int, int, int]


def generate_random_anomalies(
    *,
    num_entities: int,
    num_relations: int,
    real_triple_set: Set[Triple],
    target_pool_size: int,
    random_generator: np.random.Generator,
    max_attempts_per_sample: int = 50,
    max_total_attempts_multiplier: int = 20,
) -> List[Triple]:
    """Sample `target_pool_size` unique synthetic triples not in `real_triple_set`.

    Each candidate triple's three components are drawn uniformly from
    `range(num_entities)` for head and tail and from `range(num_relations)`
    for the relation. Candidates are rejected if:
      * the triple is already in `real_triple_set` (real-graph collision),
      * head == tail (self-loop - matches TRIC's behaviour),
      * the triple is already in the pool (would reduce uniqueness).

    Parameters
    ----------
    num_entities
        Size of the entity vocabulary (should be the UNION vocab from
        `load_kg_union`).
    num_relations
        Size of the relation vocabulary (UNION vocab).
    real_triple_set
        The set of all real `(h, r, t)` index tuples - filter against this
        to avoid false negatives. Use `kg.triple_set_idx` from
        `load_kg_union`.
    target_pool_size
        Number of unique synthetic triples to produce.
    random_generator
        `numpy.random.Generator` - pass the same seed for reproducible runs.
    max_attempts_per_sample
        Per-triple retry budget. If we can't find a valid candidate in
        this many tries we move on (avoids hangs in degenerate small
        vocabularies).
    max_total_attempts_multiplier
        Hard upper bound on total sampling attempts, expressed as a
        multiplier of `target_pool_size`. Protects against impossible
        targets (e.g. asking for more unique triples than the space
        contains minus the real triples).

    Returns
    -------
    pool : list[Triple]
        `target_pool_size` unique synthetic triples - unless the search
        space was too small, in which case as many as could be found.
    """
    pool: List[Triple] = []
    pool_set: Set[Triple] = set()
    total_attempts = 0
    total_attempts_budget = target_pool_size * max_total_attempts_multiplier

    while len(pool) < target_pool_size and total_attempts < total_attempts_budget:
        for _ in range(max_attempts_per_sample):
            total_attempts += 1
            if total_attempts > total_attempts_budget:
                break
            head_index     = int(random_generator.integers(num_entities))
            tail_index     = int(random_generator.integers(num_entities))
            relation_index = int(random_generator.integers(num_relations))
            if head_index == tail_index:
                continue
            candidate: Triple = (head_index, relation_index, tail_index)
            if candidate in real_triple_set:
                continue
            if candidate in pool_set:
                continue
            pool.append(candidate)
            pool_set.add(candidate)
            break

    return pool
