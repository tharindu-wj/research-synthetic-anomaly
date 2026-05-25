"""Distribution-match validation: how well does the trained GAN reproduce
the TRIC rule's behaviour on average?

We run the GAN and the rule each `num_samples` times on the same clean KG
and compare three things:

  1. "Edges that changed channel" - how many triples per sample left their
     original (h, r, t) coordinates? For pure change_relation, the answer
     should be ~num_corruption_steps every time.
  2. "Net edge delta" - did the corruption add or remove edges on net? For
     pure change_relation, the answer should be 0 (each swap removes one
     and adds one).
  3. Operation-mix tally for the GAN - how many of its edits look like
     change_relation vs swap_subject vs multi_op?

Both the rule baseline and the GAN are run on triple lists. Set differences
replace adjacency diffs.
"""
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import torch

from inference.two_stage import generate_k_corrupted_kgs
from .operation_mix import categorise_corruption


# (head_index, relation_index, tail_index)
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
    num_corruption_steps: int = 2,
    latent_dim: int = 8,
    gumbel_temperature: float = 0.5,
    device: torch.device | str = "cpu",
    random_seed: int = 12345,
) -> Dict:
    """Compare the GAN's corruption distribution to a rule's distribution.

    Parameters
    ----------
    generator_model, entity_embedding, relation_embedding
        The trained GAN components.
    clean_triples
        The clean KG to corrupt, as a list of (h, r, t) integer triples.
    rule_corruption_fn
        Triple-level TRIC, typically pre-bound with functools.partial.
        Signature: ``(triples, num_corruption_steps, rng) -> (corrupted_triples, edit_records)``.
    num_entities, num_relations
        Pass-through to the GAN's two-stage inference.
    num_samples
        How many comparison rounds to run. Each round = 1 rule corruption +
        1 GAN corruption.
    num_corruption_steps
        How many triples to corrupt per sample.
    latent_dim, gumbel_temperature, device
        Forwarded to the GAN inference.
    random_seed
        Seed for the numpy generator used by both the rule and the GAN's
        Stage-A selection.

    Returns
    -------
    dict with keys:
      rule_shifted_counts : (num_samples,) np.ndarray of ints
      gan_shifted_counts  : (num_samples,) np.ndarray of ints
      gan_net_delta       : (num_samples,) np.ndarray of ints
      gan_op_counts       : dict of operation -> total count across all samples
    """
    random_generator = np.random.default_rng(random_seed)
    # For set-difference comparisons against the clean KG, precompute the set
    clean_triple_set: set = set(clean_triples)

    rule_edges_shifted_per_sample: list = []
    gan_edges_shifted_per_sample:  list = []
    gan_net_edge_delta_per_sample: list = []
    gan_operation_totals = dict(
        change_relation=0, swap_subject=0, swap_object=0,
        multi_op=0, no_change=0,
    )

    total_corruption_events = num_samples * num_corruption_steps
    generator_model.eval()

    for _ in range(num_samples):
        # ── Rule baseline (triple-level TRIC) ─────────────────────────
        rule_corrupted_triples, _edits = rule_corruption_fn(
            list(clean_triples), num_corruption_steps, random_generator,
        )
        rule_corrupted_set = set(rule_corrupted_triples)
        # "Edges that changed channel" = # triples in clean but not in corrupted.
        # For pure change_relation, that equals num_corruption_steps (each
        # change_relation removes one edge and adds another at a new channel).
        rule_edges_shifted_per_sample.append(
            len(clean_triple_set - rule_corrupted_set)
        )

        # ── GAN corruption (one sample via the same two-stage pipeline) ─
        gan_corrupted_triples = generate_k_corrupted_kgs(
            generator_model, entity_embedding, relation_embedding,
            list(clean_triples),
            K_samples=1,
            num_corruption_steps=num_corruption_steps,
            latent_dim=latent_dim,
            gumbel_temperature=gumbel_temperature,
            device=device,
            random_generator=random_generator,
        )[0]
        gan_corrupted_set = set(gan_corrupted_triples)
        gan_edges_shifted_per_sample.append(
            len(clean_triple_set - gan_corrupted_set)
        )
        gan_net_edge_delta_per_sample.append(
            len(gan_corrupted_set) - len(clean_triple_set)
        )

        # Operation-mix tally for the GAN
        per_sample_op_counts = categorise_corruption(
            list(clean_triples), gan_corrupted_triples,
        )
        for operation_name in gan_operation_totals:
            gan_operation_totals[operation_name] += per_sample_op_counts[operation_name]

    rule_edges_shifted = np.array(rule_edges_shifted_per_sample)
    gan_edges_shifted  = np.array(gan_edges_shifted_per_sample)
    gan_net_edge_delta = np.array(gan_net_edge_delta_per_sample)

    print(f"Over {num_samples} samples "
          f"(each Stage-A selects {num_corruption_steps} triples):")
    print(f"  Rule - edges shifted channel: "
          f"mean={rule_edges_shifted.mean():5.1f}, "
          f"std={rule_edges_shifted.std():.2f}  (expect ~{num_corruption_steps})")
    print(f"  GAN  - edges shifted channel: "
          f"mean={gan_edges_shifted.mean():5.1f},  "
          f"std={gan_edges_shifted.std():.2f}")
    print(f"  GAN  - net edge delta:        "
          f"mean={gan_net_edge_delta.mean():+.2f}, "
          f"std={gan_net_edge_delta.std():.2f}  "
          f"(expect ~0 for change_relation)")
    print()
    print(f"GAN operation-mix across {total_corruption_events} corruption events:")
    for operation_name, count in gan_operation_totals.items():
        if operation_name == "no_change":
            continue   # this is the count of unchanged triples, not corruption events
        percentage = 100.0 * count / max(total_corruption_events, 1)
        print(f"  {operation_name:<20s}: {count:5d}  ({percentage:.1f}%)")

    return {
        "rule_shifted_counts": rule_edges_shifted,
        "gan_shifted_counts":  gan_edges_shifted,
        "gan_net_delta":       gan_net_edge_delta,
        "gan_op_counts":       gan_operation_totals,
    }
