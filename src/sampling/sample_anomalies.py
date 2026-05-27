"""Sample a deduplicated pool of synthetic anomalous triples from a trained GAN.

The TripleGenerator is clean-triple-conditioned: each forward pass takes
ONE real triple plus a noise vector and emits ONE candidate corruption.
To produce a flat pool of 400k unique non-real triples, we iterate over
the real triples, take K samples per triple, and filter against:

  * `kg.triple_set_idx`  - real-graph collisions (would be false negatives
                            from ADKGD's perspective);
  * self-loops (h == t);
  * the running `seen` set (uniqueness, mode-collapse guard).

If a single pass over the real triples isn't enough to reach the target
(common at the typical samples_per_triple=2 setting when many candidates
get rejected), we re-shuffle and run further passes up to `max_passes`.

For tractability on FB15k-237 we batch the forward pass: instead of
one-clean-triple-per-pass, we run `batch_size` clean triples at once and
harvest a full batch of candidates per generator call.

Decoding strategy
-----------------
The training signal (CE against TRIC substitution targets) has a built-in
bias: across substitution edits, 2 of the 3 target columns equal the
corresponding clean columns. So the model's modal argmax tends toward
"identity" - it outputs the input verbatim. With argmax decoding that's
a real-graph collision on every sample and the pool fills at rate 0.

Fix: per sample, pick ONE of the three positions uniformly at random,
mask out the clean index at THAT position's logits (set to -inf), then
argmax (with Gumbel noise so different z values give different outputs).
The other two positions copy the clean input. This exactly mirrors TRIC's
substitution semantics (`change_relation`, `swap_*_same_type`,
`swap_*_diff_type` all change exactly one of h/r/t) and turns every
forward pass into a guaranteed-non-identity candidate.

Set `masked_single_position=False` for vanilla argmax decoding (useful
for diagnostic comparisons; expect very low pool yields).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from models.triple_gan import (
    EntityEmbedding,
    RelationEmbedding,
    TripleGenerator,
    build_triple_embedding,
)


Triple = Tuple[int, int, int]


def _argmax_with_gumbel_and_input_mask(
    logits: torch.Tensor,
    clean_indices: torch.Tensor,
    gumbel_temperature: float,
    mask_clean_index: bool,
) -> torch.Tensor:
    """Argmax over `logits` with two adjustments.

    1. Add Gumbel noise scaled by `gumbel_temperature` so per-call argmax
       varies with the latent z (otherwise the model collapses to its
       modal prediction every call).
    2. If `mask_clean_index` is True, set the logit at the clean input
       index to -inf so argmax can never return the input. Used for the
       "guaranteed-corruption" sampling strategy described in the module
       docstring.
    """
    perturbed = logits.clone()
    if mask_clean_index:
        perturbed.scatter_(1, clean_indices.unsqueeze(1), float('-inf'))
    if gumbel_temperature > 0:
        uniform_noise = torch.rand_like(perturbed).clamp_(1e-10, 1.0 - 1e-10)
        gumbel_noise  = -torch.log(-torch.log(uniform_noise))
        perturbed = perturbed + gumbel_noise * gumbel_temperature
    return perturbed.argmax(dim=-1)


def sample_anomaly_pool(
    *,
    generator: TripleGenerator,
    entity_embedding: EntityEmbedding,
    relation_embedding: RelationEmbedding,
    clean_triples: List[Triple],
    real_triple_set: Set[Triple],
    target_pool_size: int,
    samples_per_triple: int = 2,
    max_passes: int = 5,
    batch_size: int = 256,
    gumbel_temperature: float = 0.5,
    masked_single_position: bool = True,
    device: Optional[torch.device] = None,
    random_seed: int = 0,
) -> List[Triple]:
    """Sample synthetic anomalies until the deduplicated pool hits `target_pool_size`.

    Returns the pool as a list of `(h, r, t)` index tuples. If we hit
    `max_passes` without reaching the target, returns whatever was
    collected.
    """
    if device is None:
        device = next(generator.parameters()).device

    # Switch to eval mode: deactivates dropout, etc. The Gumbel sampler
    # still adds noise (that's where the per-call variation comes from).
    generator.eval()
    entity_embedding.eval()
    relation_embedding.eval()

    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    pool: List[Triple] = []
    seen: Set[Triple] = set()
    clean_triples_array = np.asarray(clean_triples, dtype=np.int64)

    pass_index = 0
    while len(pool) < target_pool_size and pass_index < max_passes:
        pass_index += 1
        shuffle_order = rng.permutation(len(clean_triples_array))
        shuffled_clean_triples = clean_triples_array[shuffle_order]
        print(
            f"[sample] pass {pass_index}/{max_passes} - pool {len(pool):,}/{target_pool_size:,}",
            flush=True,
        )

        for batch_start in range(0, len(shuffled_clean_triples), batch_size):
            batch_end = batch_start + batch_size
            clean_batch_numpy = shuffled_clean_triples[batch_start:batch_end]
            clean_batch_tensor = torch.from_numpy(clean_batch_numpy).to(device)
            current_batch_size = clean_batch_tensor.size(0)

            clean_head = clean_batch_tensor[:, 0]
            clean_rel  = clean_batch_tensor[:, 1]
            clean_tail = clean_batch_tensor[:, 2]

            for _ in range(samples_per_triple):
                with torch.no_grad():
                    clean_embedding = build_triple_embedding(
                        clean_batch_tensor, entity_embedding, relation_embedding,
                    )
                    latent_sample = torch.randn(
                        current_batch_size, generator.latent_dim, device=device,
                    )
                    gen_output = generator(
                        clean_embedding, latent_sample,
                        gumbel_temperature=gumbel_temperature,
                        return_soft_samples=False,
                    )

                if masked_single_position:
                    # Pick one of {head=0, rel=1, tail=2} per item, mask
                    # the clean index at that position, then argmax. Other
                    # positions copy the input.
                    positions_to_corrupt = torch.from_numpy(
                        rng.integers(3, size=current_batch_size)
                    ).to(device)

                    output_heads = clean_head.clone()
                    output_rels  = clean_rel.clone()
                    output_tails = clean_tail.clone()

                    is_head_position = (positions_to_corrupt == 0)
                    is_rel_position  = (positions_to_corrupt == 1)
                    is_tail_position = (positions_to_corrupt == 2)
                    if is_head_position.any():
                        output_heads[is_head_position] = _argmax_with_gumbel_and_input_mask(
                            gen_output['head_entity_logits'][is_head_position],
                            clean_head[is_head_position],
                            gumbel_temperature, mask_clean_index=True,
                        )
                    if is_rel_position.any():
                        output_rels[is_rel_position] = _argmax_with_gumbel_and_input_mask(
                            gen_output['relation_logits'][is_rel_position],
                            clean_rel[is_rel_position],
                            gumbel_temperature, mask_clean_index=True,
                        )
                    if is_tail_position.any():
                        output_tails[is_tail_position] = _argmax_with_gumbel_and_input_mask(
                            gen_output['tail_entity_logits'][is_tail_position],
                            clean_tail[is_tail_position],
                            gumbel_temperature, mask_clean_index=True,
                        )
                else:
                    # Vanilla decoding: argmax of all three heads independently
                    # (this is what produced empty pools on the dummy KG smoke).
                    output_heads = _argmax_with_gumbel_and_input_mask(
                        gen_output['head_entity_logits'], clean_head,
                        gumbel_temperature, mask_clean_index=False,
                    )
                    output_rels = _argmax_with_gumbel_and_input_mask(
                        gen_output['relation_logits'], clean_rel,
                        gumbel_temperature, mask_clean_index=False,
                    )
                    output_tails = _argmax_with_gumbel_and_input_mask(
                        gen_output['tail_entity_logits'], clean_tail,
                        gumbel_temperature, mask_clean_index=False,
                    )

                head_numpy = output_heads.cpu().numpy()
                rel_numpy  = output_rels.cpu().numpy()
                tail_numpy = output_tails.cpu().numpy()
                for h, r, t in zip(head_numpy, rel_numpy, tail_numpy):
                    candidate = (int(h), int(r), int(t))
                    if candidate[0] == candidate[2]:
                        continue
                    if candidate in real_triple_set:
                        continue
                    if candidate in seen:
                        continue
                    seen.add(candidate)
                    pool.append(candidate)
                    if len(pool) >= target_pool_size:
                        return pool

    return pool


def load_checkpoint(
    checkpoint_path,
    *,
    device: torch.device,
):
    """Reconstruct generator + embeddings from a saved checkpoint.

    Returns a dict matching `train_triple_gan`'s output (minus `history`).
    """
    from training.train_triple_gan import TrainConfig  # local to avoid cycle

    payload: Dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: TrainConfig = payload['config']
    num_entities  = payload['num_entities']
    num_relations = payload['num_relations']

    entity_embedding   = EntityEmbedding(num_entities, config.embedding_dim).to(device)
    relation_embedding = RelationEmbedding(num_relations, config.embedding_dim).to(device)
    generator = TripleGenerator(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=config.embedding_dim,
        latent_dim=config.latent_dim,
        encoder_base_channels=config.encoder_base_channels,
        encoder_num_conv_layers=config.encoder_num_conv_layers,
        encoder_num_attn_heads=config.encoder_num_attn_heads,
        bottleneck_hidden_dim=config.bottleneck_hidden_dim,
    ).to(device)

    generator.load_state_dict(payload['generator_state_dict'])
    entity_embedding.load_state_dict(payload['entity_embedding_state_dict'])
    relation_embedding.load_state_dict(payload['relation_embedding_state_dict'])

    return {
        'generator':          generator,
        'entity_embedding':   entity_embedding,
        'relation_embedding': relation_embedding,
        'config':             config,
        'epoch':              payload.get('epoch'),
    }
