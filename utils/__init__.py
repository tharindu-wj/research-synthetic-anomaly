"""Backward-compatibility shim — imports from the new root-level packages.

Use the canonical package imports in new code:
    from build_graphs import build_dummy_kg, visualise_kg_subgraph, ...
    from corruption_strategies import apply_kg_edge_removal, apply_tric_corruption
    from dataset_builders import build_kg_paired_dataset
"""

from build_graphs import (
    build_dummy_kg,
    get_dummy_kg_metadata,
    find_well_connected_ego_entity,
    build_kg_subgraph,
    summarise_kg_subgraph,
    infer_entity_type_labels,
    visualise_kg_subgraph,
)
from corruption_strategies import apply_kg_edge_removal, apply_tric_corruption
from dataset_builders import build_kg_paired_dataset

__all__ = [
    "build_dummy_kg",
    "get_dummy_kg_metadata",
    "find_well_connected_ego_entity",
    "build_kg_subgraph",
    "summarise_kg_subgraph",
    "infer_entity_type_labels",
    "visualise_kg_subgraph",
    "apply_kg_edge_removal",
    "apply_tric_corruption",
    "build_kg_paired_dataset",
]
