"""Triple list -> (N, N, R) adjacency tensor."""
from typing import Dict, Iterable, Tuple
import numpy as np


def triples_to_adjacency_tensor(
    triples: Iterable[Tuple[str, str, str]],
    entity_id_to_row: Dict[str, int],
    relation_id_to_channel: Dict[str, int],
) -> np.ndarray:
    """Build a (N, N, R) float32 adjacency tensor from a list of triples.

    Parameters
    ----------
    triples                : iterable of (head_id, relation_id, tail_id) strings
    entity_id_to_row       : str -> 0..N-1
    relation_id_to_channel : str -> 0..R-1

    Returns
    -------
    np.ndarray of shape (N, N, R), float32, values in {0.0, 1.0}.
    A[i, j, r] = 1 iff (entities[i], relations[r], entities[j]) is a triple.
    """
    N = len(entity_id_to_row)
    R = len(relation_id_to_channel)
    adj = np.zeros((N, N, R), dtype=np.float32)
    for h, r, t in triples:
        if h not in entity_id_to_row or t not in entity_id_to_row:
            continue
        if r not in relation_id_to_channel:
            continue
        adj[entity_id_to_row[h], entity_id_to_row[t], relation_id_to_channel[r]] = 1.0
    return adj
