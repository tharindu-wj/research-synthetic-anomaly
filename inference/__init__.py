"""Inference utilities: turn a trained triple-level generator into a graph-level corruptor."""

from .two_stage import generate_k_corrupted_kgs, kg_triples_to_nxnxr

__all__ = [
    "generate_k_corrupted_kgs",
    "kg_triples_to_nxnxr",
]
