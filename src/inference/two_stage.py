"""Two-stage inference: turn a per-triple generator into a graph-level corruptor.

The trained generator only knows how to corrupt ONE triple at a time. At
inference, we want to corrupt a whole KG (a set of triples). So we wrap the
generator in a two-stage process:

  Stage A (selection):
      Randomly pick which triples to corrupt. This is a simple rule (not
      learned) - we mimic TRIC's behaviour of choosing `num_corruption_steps`
      triples from the clean KG.

  Stage B (transformation):
      For each picked triple, sample a latent vector `z`, run the generator,
      and take argmax of the three softmax heads to get (h', r', t').

Running the whole pipeline K times with different randomness yields K
distinct corrupted versions of the same clean KG.
"""
from typing import List, Sequence, Tuple

import numpy as np
import torch


# (head_index, relation_index, tail_index)
Triple = Tuple[int, int, int]


def generate_k_corrupted_kgs(
    generator_model: torch.nn.Module,
    entity_embedding: torch.nn.Module,
    relation_embedding: torch.nn.Module,
    clean_triples: Sequence[Triple],
    *,
    K_samples: int = 20,
    num_corruption_steps: int = 2,
    latent_dim: int = 8,
    gumbel_temperature: float = 0.5,
    device: torch.device | str = "cpu",
    random_generator: np.random.Generator | None = None,
) -> List[List[Triple]]:
    """Generate K distinct corrupted versions of the clean KG.

    Parameters
    ----------
    generator_model
        The trained TripleGenerator. Must accept `(clean_triple_embedding,
        latent_sample, gumbel_temperature, return_soft)` and return a dict
        with keys "h_logits", "r_logits", "t_logits".
    entity_embedding, relation_embedding
        The trained learnable lookup tables. Calling them with a 1-D tensor
        of indices returns the corresponding (M, embedding_dim) tensor.
    clean_triples
        The KG to corrupt, as a list of (head_index, relation_index, tail_index).
    K_samples
        How many distinct corrupted KGs to generate. Each one is a fresh
        Stage-A selection + per-triple Stage-B forward pass.
    num_corruption_steps
        How many triples per sample to corrupt. The rest pass through unchanged.
    latent_dim
        Dimensionality of the latent z. Must match what the generator expects.
    gumbel_temperature
        Inverse-temperature for the Gumbel-Softmax (only used by the generator
        if return_soft=True; here we use argmax so it doesn't matter much).
    device
        Where to run the forward pass. Embeddings + model + tensors must
        all be on the same device.
    random_generator
        Optional numpy Generator for the Stage-A random selection. Pass one
        for reproducibility.

    Returns
    -------
    list of length K_samples. Each element is a list[Triple] of length
    len(clean_triples) - one entry per original triple, with corrupted
    triples replacing the selected positions.
    """
    if random_generator is None:
        random_generator = np.random.default_rng()
    num_clean_triples         = len(clean_triples)
    num_triples_to_corrupt    = min(num_corruption_steps, num_clean_triples)

    # Switch the generator into eval mode (disables dropout, etc.) and turn
    # off autograd: we're just sampling, not training.
    generator_model.eval()
    all_corrupted_kgs: List[List[Triple]] = []

    with torch.no_grad():
        for _ in range(K_samples):
            # ── Stage A: pick which triples to corrupt ────────────────
            selected_triple_indices = set(
                int(picked_index)
                for picked_index in random_generator.choice(
                    num_clean_triples, size=num_triples_to_corrupt, replace=False,
                )
            )

            # ── Stage B: per-triple G forward (and pass-through the rest) ─
            corrupted_kg: List[Triple] = []
            for position_index, (head_index, relation_index, tail_index) in enumerate(clean_triples):
                if position_index not in selected_triple_indices:
                    # Unselected -> emit unchanged
                    corrupted_kg.append((head_index, relation_index, tail_index))
                    continue

                # Look up the three embeddings for this single triple
                head_embedding_vector     = entity_embedding(torch.tensor([head_index],     device=device))
                relation_embedding_vector = relation_embedding(torch.tensor([relation_index], device=device))
                tail_embedding_vector     = entity_embedding(torch.tensor([tail_index],     device=device))
                # Stack into (1, 3, embedding_dim) - the generator's input shape
                clean_triple_embedding = torch.stack(
                    [head_embedding_vector, relation_embedding_vector, tail_embedding_vector],
                    dim=1,
                )

                # Sample a fresh latent z for this triple
                latent_sample = torch.randn(1, latent_dim, device=device)

                # Forward pass through the generator. return_soft_samples=False
                # means we get only the raw logits (we'll argmax them next).
                generator_output = generator_model(
                    clean_triple_embedding, latent_sample,
                    gumbel_temperature=gumbel_temperature, return_soft_samples=False,
                )
                # argmax over each vocabulary to get a concrete triple
                new_head_index     = int(generator_output["head_entity_logits"].argmax(dim=-1).item())
                new_relation_index = int(generator_output["relation_logits"].argmax(dim=-1).item())
                new_tail_index     = int(generator_output["tail_entity_logits"].argmax(dim=-1).item())
                corrupted_kg.append((new_head_index, new_relation_index, new_tail_index))

            all_corrupted_kgs.append(corrupted_kg)

    return all_corrupted_kgs


def kg_triples_to_nxnxr(
    triples: Sequence[Triple], num_nodes: int, num_relations: int,
) -> np.ndarray:
    """Convert a list of (h, r, t) index triples to an (N, N, R) adjacency tensor.

    Used by the visualisation code, which wants an adjacency tensor to draw.
    Self-loops (h == t) are skipped.

    Parameters
    ----------
    triples       : iterable of (head_index, relation_index, tail_index) tuples
    num_nodes     : entity vocabulary size N
    num_relations : relation vocabulary size R

    Returns
    -------
    np.ndarray of shape (N, N, R), float32, values in {0.0, 1.0}.
    """
    adjacency_tensor = np.zeros((num_nodes, num_nodes, num_relations), dtype=np.float32)
    for head_index, relation_index, tail_index in triples:
        if head_index != tail_index:
            adjacency_tensor[head_index, tail_index, relation_index] = 1.0
    return adjacency_tensor
