"""Convert a list of triples into an (N, N, R) adjacency tensor.

The pipeline as a whole uses triple lists, NOT adjacency. The only places
that want adjacency are the visualisation helpers (and that's because
NetworkX prefers it). We build it on demand here so the dataflow stays
triple-only.

For the dummy KG (N=10, R=3): tensor is tiny (~1 KB).
For FB15k-237 (N=14.5k, R=237): the tensor would be ~199 GB - DON'T call this
function on that scale. Use a subgraph instead.
"""
from typing import Dict, Iterable, Tuple

import numpy as np


def triples_to_adjacency_tensor(
    triples: Iterable[Tuple[str, str, str]],
    entity_id_to_row: Dict[str, int],
    relation_id_to_channel: Dict[str, int],
) -> np.ndarray:
    """Build a binary (N, N, R) adjacency tensor from a list of string triples.

    Parameters
    ----------
    triples
        Iterable of (head_id, relation_id, tail_id) string triples (the
        "raw" form, the one a loader produces from a TSV file).
    entity_id_to_row
        Map from entity string id (e.g. "/dummy/Alice") to its row index
        (0 .. num_entities - 1).
    relation_id_to_channel
        Map from relation string id (e.g. "/dummy/born_in") to its channel
        index (0 .. num_relations - 1).

    Returns
    -------
    np.ndarray of shape (N, N, R), dtype float32, values in {0.0, 1.0}.

    `adjacency[head_row, tail_row, relation_channel] = 1.0` iff that triple
    exists in the input.
    """
    num_entities  = len(entity_id_to_row)
    num_relations = len(relation_id_to_channel)
    adjacency_tensor = np.zeros((num_entities, num_entities, num_relations), dtype=np.float32)

    for head_string, relation_string, tail_string in triples:
        # Skip triples whose entities or relation aren't in the vocab
        # (defensive: shouldn't happen if `triples` came from a loaded KG).
        if head_string not in entity_id_to_row or tail_string not in entity_id_to_row:
            continue
        if relation_string not in relation_id_to_channel:
            continue
        head_row     = entity_id_to_row[head_string]
        tail_row     = entity_id_to_row[tail_string]
        relation_ch  = relation_id_to_channel[relation_string]
        adjacency_tensor[head_row, tail_row, relation_ch] = 1.0
    return adjacency_tensor
