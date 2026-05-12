"""CorruptionStrategy protocol — any callable matching this signature is a valid strategy."""
from typing import Protocol
import numpy as np


class CorruptionStrategy(Protocol):
    """A corruption strategy takes a (N, N, R) adjacency tensor and returns a corrupted copy."""

    def __call__(
        self,
        adj_nxnxr: np.ndarray,
        num_corruptions: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...
