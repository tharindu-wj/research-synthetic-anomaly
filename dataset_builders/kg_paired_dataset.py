"""Paired (clean, corrupted) dataset builder for multi-relational KG GAN training."""
import numpy as np
import torch

from corruption_strategies.edge_removal import apply_kg_edge_removal


def build_kg_paired_dataset(
    adj_tensor: np.ndarray,
    num_pairs: int,
    num_corruptions: int,
    rng: np.random.Generator,
    corruption_fn=None,
):
    """Build a (clean, corrupted) paired dataset for multi-relational KG GAN training.

    Mirrors build_paired_dataset() from the Facebook pipeline, adapted for the
    (N, N, R) tensor format and directed (non-symmetric) adjacency.

    Parameters
    ----------
    adj_tensor     : np.ndarray, shape (N, N, R), float32
        Source adjacency tensor. Directed — no symmetry assumed. Diagonal = 0.
    num_pairs      : int
    num_corruptions: int
        Number of corruption operations per sample. Passed directly to
        corruption_fn as its second positional argument.
    rng            : np.random.Generator
    corruption_fn  : callable or None
        A corruption strategy with signature::

            corrupted = corruption_fn(adj_nxnxr, num_corruptions, rng)

        Defaults to apply_kg_edge_removal (simple random edge deletion).
        Use functools.partial to pre-bind extra keyword arguments, e.g.::

            import functools
            from corruption_strategies import apply_tric_corruption
            fn = functools.partial(apply_tric_corruption, node_types=node_types)
            clean, corrupted, perms = build_kg_paired_dataset(..., corruption_fn=fn)

    Returns
    -------
    clean_pairs     : torch.Tensor, shape (B, R, N, N), float32, values in [-1, 1]
    corrupted_pairs : torch.Tensor, shape (B, R, N, N), float32, values in [-1, 1]
    permutations    : torch.Tensor, shape (B, N), int64

    Notes
    -----
    Permutation is applied uniformly to BOTH spatial axes (rows and columns) across
    ALL R slices. The R axis is NEVER permuted — relation identity is semantic,
    not structural.

    Values are rescaled from {0, 1} to [-1, 1] (tanh/DCGAN convention) so the
    generator's logit-space residual sees a consistent input range.
    """
    if corruption_fn is None:
        corruption_fn = apply_kg_edge_removal

    N, _, R = adj_tensor.shape
    cleans, corrupted_list, perm_list = [], [], []

    for _ in range(num_pairs):
        pi = rng.permutation(N)
        A      = adj_tensor[np.ix_(pi, pi)]           # (N, N, R) — permute rows & cols
        A_corr = corruption_fn(A, num_corruptions, rng)

        # Transpose to channel-first (R, N, N) and rescale [0,1] → [-1, 1]
        cleans.append(np.transpose(A,      (2, 0, 1)) * 2.0 - 1.0)
        corrupted_list.append(np.transpose(A_corr, (2, 0, 1)) * 2.0 - 1.0)
        perm_list.append(pi)

    clean     = torch.tensor(np.stack(cleans),         dtype=torch.float32)
    corrupted = torch.tensor(np.stack(corrupted_list), dtype=torch.float32)
    perms     = torch.tensor(np.stack(perm_list),      dtype=torch.long)
    return clean, corrupted, perms
