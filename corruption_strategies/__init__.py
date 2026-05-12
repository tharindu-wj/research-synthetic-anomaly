"""Corruption strategies for multi-relational KG adjacency tensors.

Available strategies
--------------------
apply_kg_edge_removal   Simple uniform random edge deletion.
apply_tric_corruption   TRIC-style semantic corruption (Senaratne et al., ESWC 2023).

All strategies share the same call signature::

    corrupted = strategy(adj_nxnxr, num_corruptions, rng)

Extra keyword arguments (e.g. node_types for TRIC) can be pre-bound using
functools.partial before passing to build_kg_paired_dataset.
"""

from .edge_removal import apply_kg_edge_removal
from .tric import apply_tric_corruption

__all__ = [
    "apply_kg_edge_removal",
    "apply_tric_corruption",
]
