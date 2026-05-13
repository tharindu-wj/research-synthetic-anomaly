"""Unified KG loading for dummy and FB15k-237-format datasets.

Typical use::

    from kg_data import load_kg
    kg = load_kg('datasets/dummy_kg')

    # Now use:
    kg.adjacency_tensor      # (N, N, R) np.float32
    kg.entity_id_to_row      # str -> int
    kg.relation_id_to_channel  # str -> int
    kg.node_labels           # int -> str (display names)
    kg.node_types            # int -> str (entity types; may be 'unknown')
    kg.relation_names        # list[str], indexed by channel
"""

from .loader import KnowledgeGraph, load_kg
from .adjacency import triples_to_adjacency_tensor
from .dummy_kg_writer import write_dummy_kg

__all__ = [
    "KnowledgeGraph",
    "load_kg",
    "triples_to_adjacency_tensor",
    "write_dummy_kg",
]
