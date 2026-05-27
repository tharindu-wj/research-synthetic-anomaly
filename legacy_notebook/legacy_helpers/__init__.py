"""Helpers used exclusively by the frozen research notebook.

Originally lived under `src/kg_data/`. They were moved here when the
ADKGD pipeline was carved out (see `legacy_notebook/README.md`). The
notebook keeps using them via::

    from legacy_helpers import triples_to_adjacency_tensor, write_dummy_kg
"""

from .adjacency import triples_to_adjacency_tensor
from .dummy_kg_writer import write_dummy_kg

__all__ = [
    "triples_to_adjacency_tensor",
    "write_dummy_kg",
]
