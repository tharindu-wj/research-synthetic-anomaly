"""Distribution-match validation: how well does the GAN reproduce the TRIC rule?

Both the rule baseline and the GAN are run on triple lists. Adjacency is built
on the fly only to count "edges that changed channel" (a graph-level metric).
"""
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from inference.two_stage import generate_k_corrupted_kgs, kg_triples_to_nxnxr
from kg_data.adjacency import triples_to_adjacency_tensor   # noqa: F401  (kept for callers)
from .operation_mix import categorise_corruption


Triple = Tuple[int, int, int]


def validate_distribution(
    *,
    generator_model: torch.nn.Module,
    entity_embedding: torch.nn.Module,
    relation_embedding: torch.nn.Module,
    clean_triples: Sequence[Triple],
    rule_corruption_fn: Callable,
    num_entities: int,
    num_relations: int,
    num_samples: int = 200,
    num_corruptions: int = 2,
    latent_dim: int = 8,
    tau: float = 0.5,
    device: torch.device | str = "cpu",
    seed: int = 12345,
) -> Dict:
    """Compare GAN corruption distribution to a rule's distribution.

    `rule_corruption_fn` is the triple-level TRIC: `(triples, num_corruptions,
    rng) -> (corrupted_triples, edit_records)`. Pre-bind kwargs via
    functools.partial.

    Returns dict with mean/std for rule edges-shifted, GAN edges-shifted,
    GAN net-edge-delta, plus a per-operation count tally.
    """
    rng = np.random.default_rng(seed)
    clean_set = set(clean_triples)

    rule_shifted_counts = []
    gan_shifted_counts  = []
    gan_net_delta       = []
    gan_op_counts       = dict(change_relation=0, swap_subject=0, swap_object=0,
                               multi_op=0, no_change=0)

    total_corruption_events = num_samples * num_corruptions
    generator_model.eval()

    for _ in range(num_samples):
        # ----- Rule corruption (triple-level) -----
        rule_corr_triples, _edits = rule_corruption_fn(
            list(clean_triples), num_corruptions, rng,
        )
        rule_corr_set = set(rule_corr_triples)
        # "edges that changed channel" = removed-from-clean edges = clean - corrupted
        rule_shifted_counts.append(len(clean_set - rule_corr_set))

        # ----- GAN corruption (one sample via two-stage pipeline) -----
        out_kg = generate_k_corrupted_kgs(
            generator_model, entity_embedding, relation_embedding,
            list(clean_triples), K=1, num_corruptions=num_corruptions,
            latent_dim=latent_dim, tau=tau, device=device, rng=rng,
        )[0]
        gan_corr_set = set(out_kg)
        gan_shifted_counts.append(len(clean_set - gan_corr_set))
        gan_net_delta.append(len(gan_corr_set) - len(clean_set))

        c = categorise_corruption(list(clean_triples), out_kg)
        for key in gan_op_counts:
            gan_op_counts[key] += c[key]

    rule_shifted_counts = np.array(rule_shifted_counts)
    gan_shifted_counts  = np.array(gan_shifted_counts)
    gan_net_delta       = np.array(gan_net_delta)

    print(f"Over {num_samples} samples (each Stage-A selects {num_corruptions} triples):")
    print(f"  Rule - edges shifted channel: mean={rule_shifted_counts.mean():5.1f}, "
          f"std={rule_shifted_counts.std():.2f}  (expect ~{num_corruptions})")
    print(f"  GAN  - edges shifted channel: mean={gan_shifted_counts.mean():5.1f},  "
          f"std={gan_shifted_counts.std():.2f}")
    print(f"  GAN  - net edge delta:        mean={gan_net_delta.mean():+.2f}, "
          f"std={gan_net_delta.std():.2f}  (expect ~0 for change_relation)")
    print()
    print(f"GAN operation-mix across {total_corruption_events} corruption events:")
    for key, count in gan_op_counts.items():
        if key == "no_change":
            continue
        pct = 100.0 * count / max(total_corruption_events, 1)
        print(f"  {key:<20s}: {count:5d}  ({pct:.1f}%)")

    return {
        "rule_shifted_counts": rule_shifted_counts,
        "gan_shifted_counts":  gan_shifted_counts,
        "gan_net_delta":       gan_net_delta,
        "gan_op_counts":       gan_op_counts,
    }
