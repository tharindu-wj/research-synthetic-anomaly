# End-to-End Data Pipeline & Training — A Guide for an Intern AI Engineer

## Context

This document is a companion to the earlier Path D-Hybrid teaching doc (in
`guide.docx`). That one explained *why the model looks the way it does* (Dai +
Pix2Pix combination). This one explains the *data pipeline*: how data flows
from the TSV files on disk all the way to the trained generator, what shape
each variable has at each stage, and how the training loop uses that data.

Audience: an AI engineer intern who:

- is comfortable with Python (loops, classes, modules)
- has heard of PyTorch but isn't yet fluent
- has heard of GANs at a high level
- has just read the Path D-Hybrid doc

The pipeline is **triple-only end-to-end**. There is no adjacency tensor
anywhere in the data flow; the model is triple-level (G consumes
`(batch_size, 3, embedding_dim)` triple embeddings, D consumes triple-pair
embeddings). Adjacency is built on-the-fly *only* inside visualisation cells.

The notebook starts with a **PyTorch primer** markdown cell explaining
`nn.Module`, `nn.Embedding`, `torch.no_grad()`, `loss.backward()`, and the
other PyTorch idioms you'll meet. Read that first if you're new to PyTorch.

## 1. The pipeline in one paragraph

We have a knowledge graph stored as TSV text files. The pipeline (a) reads
the triples and assigns integer indices to entities and relations; (b)
generates many `(clean_triple, corrupted_triple)` training pairs by applying
TRIC `change_relation` corruption to permuted views of the KG; (c) trains a
triple-level GAN where the generator learns to map a clean triple to a
corrupted one; (d) at inference time, runs the trained generator on
randomly-selected triples from a clean KG to produce diverse corrupted
variants.

## 2. Big-picture diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  data/dummy_kg/                                                             │
│  ├── train.txt              /dummy/Alice /dummy/born_in /dummy/Australia... │
│  ├── entity_metadata.txt    /dummy/Alice  Alice  Person                     │
│  └── relation_metadata.txt  /dummy/born_in  born_in                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  load_kg(path)                  Stage 1
┌─────────────────────────────────────────────────────────────────────────────┐
│  knowledge_graph : KnowledgeGraph                                           │
│    triples:        List[(str, str, str)]      raw, for debugging            │
│    triples_idx:    List[(int, int, int)]      what code below consumes      │
│    triple_set_idx: Set[(int, int, int)]       O(1) edge-exists check        │
│    entity_id_to_row, relation_id_to_channel   vocab maps                    │
│    node_labels, node_types, relation_names    pretty-print helpers          │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  build_triple_pair_dataset(...)  Stage 2
┌─────────────────────────────────────────────────────────────────────────────┐
│  triple_pair_tensor : torch.long, shape (M, 6)                              │
│  Columns: [clean_h, clean_r, clean_t,  target_h, target_r, target_t]        │
│  Example row [0, 0, 1, 0, 1, 1]                                             │
│       = (Alice, born_in, Australia)  →  (Alice, married_to, Australia)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  DataLoader (batch_size=64)    Stage 3
┌─────────────────────────────────────────────────────────────────────────────┐
│  clean_triple_indices_batch  : torch.long  shape (64, 3)                    │
│  target_triple_indices_batch : torch.long  shape (64, 3)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  entity_embedding / relation_embedding  Stage 3
┌─────────────────────────────────────────────────────────────────────────────┐
│  clean_triple_embedding_for_g_step  : torch.float  shape (64, 3, 100)       │
│  target_triple_embedding_for_d_step : torch.float  shape (64, 3, 100)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  generator_model(clean, latent)  Stage 4
┌─────────────────────────────────────────────────────────────────────────────┐
│  head_entity_logits (64, 10)                                                │
│  relation_logits    (64, 3)                                                 │
│  tail_entity_logits (64, 10)                                                │
│  Gumbel-Softmax  → head_entity_soft_sample, relation_soft_sample, tail_..   │
│  candidate_embedding_from_z1 : (64, 3, 100)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  discriminator_model(clean, candidate)  Stage 5
┌─────────────────────────────────────────────────────────────────────────────┐
│  d_score_on_real_pair / d_score_on_fake_pair : shape (64, 1)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼  loss_reconstruction + loss_adversarial + loss_diversity, backward()   Stage 6
                              [training loop, 400 epochs]
                                  │
                                  ▼  generate_k_corrupted_kgs(...)   Stage 7
                              [K=20 corrupted KGs at inference time]
```

## 3. Stage 1 — Loading the KG (TSV → namespace)

**Call site (notebook Step 2):**

```python
knowledge_graph = load_kg('data/dummy_kg')
```

**Code:** [kg_data/loader.py](kg_data/loader.py).

Inside `load_kg`:

1. **Read `train.txt`** line by line; each line is a `(head_str,
   relation_str, tail_str)` tuple. 18 triples for the dummy KG.
2. **Build vocabularies** in first-seen order. Walk through the triples;
   assign row 0 to the first entity seen, row 1 to the second, and so on.
   Same for relations → channel indices.
   - Result: `entity_id_to_row: Dict[str, int]` and
     `relation_id_to_channel: Dict[str, int]`.
3. **Read metadata files** (optional). Map each entity_id to a display name
   and type; map each relation_id to a display name.
4. **Build index-form triples**: each string triple `(h_str, r_str, t_str)`
   becomes `(h_idx, r_idx, t_idx)` via the vocab maps. This is the form
   that all downstream code consumes.

**Output namespace** — fields you'll see referenced throughout the notebook
under the variable name `knowledge_graph`:

| Field | Type | Example value | Purpose |
|---|---|---|---|
| `triples` | `List[(str, str, str)]` | `[("/dummy/Alice", "/dummy/born_in", "/dummy/Australia"), ...]` | Raw form; debugging, exporting |
| `triples_idx` | `List[(int, int, int)]` | `[(0, 0, 1), (2, 0, 1), (3, 0, 4), ...]` | What downstream code consumes |
| `triple_set_idx` | `Set[(int, int, int)]` | same content as `triples_idx` | O(1) "does this edge exist?" |
| `entity_id_to_row` | `Dict[str, int]` | `{"/dummy/Alice": 0, "/dummy/Australia": 1, ...}` | String → row lookup |
| `relation_id_to_channel` | `Dict[str, int]` | `{"/dummy/born_in": 0, ...}` | String → channel lookup |
| `node_labels` | `Dict[int, str]` | `{0: "Alice", 1: "Australia", 2: "Bob", ...}` | Pretty-print a row index |
| `node_types` | `Dict[int, str]` | `{0: "Person", 1: "Country", ...}` | TRIC type-aware ops use this |
| `relation_names` | `List[str]` | `["born_in", "married_to", "lives_in"]` | Pretty-print a channel index |
| `num_nodes` (property) | `int` | 10 | |
| `num_relations` (property) | `int` | 3 | |

In the notebook Step 2 we alias these to shorter local names for convenience:

```python
num_entities         = knowledge_graph.num_nodes        # 10
num_relations        = knowledge_graph.num_relations    # 3
node_labels          = knowledge_graph.node_labels
node_types           = knowledge_graph.node_types
relation_names       = knowledge_graph.relation_names
clean_triples_in_kg  = list(knowledge_graph.triples_idx)
```

**Why two forms (string + int)?** Strings are how the data lives on disk
(and how FB15k-237 stores its triples). Integers are what `nn.Embedding`
needs as input. The vocab maps are the bridge between them.

**Why `triple_set_idx`?** TRIC needs to ask "does triple `(h, r, t)` already
exist in this KG?" hundreds of times per corruption call. A set gives O(1)
lookup; a list would be O(M). For FB15k-237 (M=272k), a list lookup would
be a real bottleneck.

**Why no adjacency tensor?** Three reasons:

- The model never reads adjacency — it reads triples.
- Adjacency is `O(N²·R)` memory; for FB15k-237 that's ~199 GB.
- Building adjacency on-demand inside visualisation cells covers the few
  places that genuinely want it.

## 4. Stage 2 — Building training pairs

**Call site (notebook Step 4):**

```python
triple_pair_tensor = build_triple_pair_dataset(
    clean_triples_in_kg,
    num_rounds=num_training_rounds,
    num_corruption_steps_per_round=num_corruption_steps_per_round,
    random_generator=training_pair_random_generator,
    num_entities=num_entities,
    num_relations=num_relations,
    corruption_fn=tric_corruption_fn,
    permute_entities=True,
)
```

**Code:** [dataset_builders/triple_pair_dataset.py](dataset_builders/triple_pair_dataset.py)
and [corruption_strategies/tric.py](corruption_strategies/tric.py).

The job of this stage: take 18 clean triples and produce ~7,640 training
pairs of the form `(clean_triple, target_triple)`.

For each of `num_training_rounds = 4000` rounds:

1. **Permute entity indices** (data augmentation). Sample a random
   permutation `entity_permutation = [3, 7, 1, 8, ...]` of length 10.
   Rewrite every clean triple:
   `(h, r, t) → (entity_permutation[h], r, entity_permutation[t])`.
   The same KG structure is now expressed with different entity IDs.

   **Why?** Without permutation the model only sees 18 distinct clean
   triples — it would memorise specific IDs. With permutation it has to
   learn the *pattern* (e.g. "if head and tail are connected by relation
   X, they can also be connected by relation Y") independent of which
   entities they are.

2. **Apply TRIC corruption** to the permuted triple list. With
   `num_corruption_steps_per_round = 2` and
   `operation_weights = {'change_relation': 1}`:
   - Pick 2 random clean triples.
   - For each, find a valid `r'` such that `(h, r', t)` is NOT already in
     the triple set.
   - Replace the triple in the working list.
   - **Record the edit** as `(clean_triple, corrupted_triple)`.

3. **Harvest substitution edits**. TRIC returns BOTH the corrupted triple
   list AND a list of edit records. We only keep records where both clean
   and corrupted are populated (pure adds/removes lack one half and can't
   be a training pair).

4. **Stack into the tensor**. Each surviving edit `(c, t)` becomes one row
   `[clean_h, clean_r, clean_t, target_h, target_r, target_t]` in the
   `(M, 6)` tensor.

**Output**: a `torch.long` tensor of shape `(M, 6)`. For
`num_training_rounds=4000, num_corruption_steps_per_round=2`, `M ≈ 7,640`
(some rounds skip ops when no valid target exists).

**Sanity check** the notebook prints:

```
change_relation sanity: same_h=1.000, same_t=1.000, diff_r=1.000  (all should be 1.0)
```

Every row has the same head index, same tail index, different relation
index — exactly what `change_relation` should produce.

## 5. Stage 3 — DataLoader and embedding lookup

**DataLoader** (notebook Step 4):

```python
training_clean_triples  = triple_pair_tensor[:, :3]
training_target_triples = triple_pair_tensor[:, 3:]

triple_pair_torch_dataset = data_utils.TensorDataset(
    training_clean_triples, training_target_triples,
)
triple_pair_dataloader = data_utils.DataLoader(
    triple_pair_torch_dataset, batch_size=64, shuffle=True, drop_last=True,
)
```

Each batch yields two `(64, 3)` long tensors:
`clean_triple_indices_batch` and `target_triple_indices_batch`.

**Embedding lookup** (inside the training loop, Step 8):

```python
def build_triple_embedding(triple_indices, entity_emb_module, relation_emb_module):
    return torch.stack([
        entity_emb_module(triple_indices[:, 0]),      # (64, 100)
        relation_emb_module(triple_indices[:, 1]),    # (64, 100)
        entity_emb_module(triple_indices[:, 2]),      # (64, 100)
    ], dim=1)

clean_triple_embedding_for_g_step = build_triple_embedding(
    clean_triple_indices_batch, entity_embedding, relation_embedding,
)
```

`entity_embedding` is an `EntityEmbedding` module wrapping
`nn.Embedding(10, 100)` — a 10 × 100 lookup table, randomly initialised,
trained jointly with the generator.

`relation_embedding` is the same idea but `(3, 100)` for relations.

After lookup, every triple `(h_idx, r_idx, t_idx)` becomes a `(3, 100)`
matrix — three 100-dim vectors stacked. Batched:
`clean_triple_embedding_for_g_step` is `(64, 3, 100)`.

The **same lookup** is also used to build `target_triple_embedding_for_d_step`
for the discriminator. The target embeddings serve as the "real" side of D's
training signal.

## 6. Stage 4 — Generator forward pass

**Call site (training loop):**

```python
generator_output_z1 = generator_model(
    clean_triple_embedding_for_g_step, latent_sample_1,
    gumbel_temperature=current_gumbel_temperature, return_soft_samples=True,
)
```

**Code:** notebook Step 6 — `TripleGenerator`.

Shape transformations through G:

```
INPUT: clean_triple_embedding_for_g_step (64, 3, 100), latent_sample_1 (64, 8)

  ┌─ ENCODER (Dai 2020 structural channel) ────────────────────────┐
  │  unsqueeze → (64, 1, 3, 100)             treat as 1-channel image│
  │  3× (Conv2d 1×1 + LeakyReLU, doubling channels)                  │
  │       encoder_conv_stack → (64, 128, 3, 100)                     │
  │  permute + reshape per token → (64, 3, 12800)                    │
  │  per_token_projection (12800 → 128) → (64, 3, 128)               │
  │  self_attention_layer (4 heads) → (64, 3, 128)                   │
  │  LayerNorm + residual → refined_tokens (64, 3, 128)              │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ BOTTLENECK (Pix2Pix-style conditional translation + latent z) ─┐
  │  flatten tokens → flattened_token_features (64, 384)             │
  │  concat with latent_sample_1  → bottleneck_input (64, 392)       │
  │  bottleneck_projection (392 → 128) → bottleneck_hidden_state (64, 128) │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ THREE OUTPUT HEADS (the structural innovation) ────────────────┐
  │  bottleneck_hidden_state (64, 128)                               │
  │   → head_entity_predictor → head_entity_logits (64, 10)          │
  │   → relation_predictor    → relation_logits    (64, 3)           │
  │   → tail_entity_predictor → tail_entity_logits (64, 10)          │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ GUMBEL-SOFTMAX (differentiable categorical sample) ────────────┐
  │  head_entity_logits → head_entity_soft_sample (64, 10)           │
  │  relation_logits    → relation_soft_sample    (64, 3)            │
  │  tail_entity_logits → tail_entity_soft_sample (64, 10)           │
  └──────────────────────────────────────────────────────────────────┘

  ┌─ RE-EMBED FOR D ────────────────────────────────────────────────┐
  │  head_entity_embedding = head_entity_soft_sample @ E_weight      │
  │                          (64, 10) @ (10, 100) = (64, 100)        │
  │  relation_embedding    = relation_soft_sample    @ R_weight      │
  │                          (64, 3)  @ (3, 100)  = (64, 100)        │
  │  tail_entity_embedding = tail_entity_soft_sample @ E_weight      │
  │                          → (64, 100)                              │
  │  candidate_embedding_from_z1 = stack(...) = (64, 3, 100)         │
  └──────────────────────────────────────────────────────────────────┘
```

**Why `head_entity_soft_sample @ entity_weight`?** Because `soft_sample` is
near-one-hot, this matrix product approximates "look up the embedding of
the most-probable entity" — but `soft_sample` is differentiable, so
autograd flows back through it to G's parameters and to the embedding
matrices.

## 7. Stage 5 — Discriminator forward pass

**Call site (training loop):**

```python
d_score_on_real_pair = discriminator_model(
    clean_with_noise_real, target_with_noise_real,
)
```

**Code:** notebook Step 6 — `TripleDiscriminator`.

```
clean_triple_embedding       (64, 3, 100)
candidate_triple_embedding   (64, 3, 100)
   ↓  flatten and concatenate
                             (64, 600)
   ↓  Linear(600 → 256) + spectral_norm + LeakyReLU
                             (64, 256)
   ↓  Linear(256 → 128) + spectral_norm + LeakyReLU
                             (64, 128)
   ↓  Linear(128 → 1)   + spectral_norm
d_score_on_real_pair         (64, 1)            real/fake logit
```

D sees **both** clean and candidate concatenated. This is what makes it a
*conditional* discriminator (the cGAN in Pix2Pix). Its judgement is "is
this corruption consistent with the clean input?", not just "does this
triple look real?".

`spectral_norm` on every linear layer caps the layer's Lipschitz constant
— a known stabiliser that prevents D from becoming too confident too fast.

## 8. Stage 6 — Loss + backward (the training step)

**Pix2Pix paradigm**: alternate G and D updates every batch.

### D-step (only updates D)

1. Build `clean_triple_embedding_for_d_step` and
   `target_triple_embedding_for_d_step` **under `torch.no_grad`** — we
   don't want gradients to flow back to E or R during D-step.
2. Forward G under `no_grad` to produce `candidate_embedding_for_d_step`
   (detached fake).
3. Compute the two D losses:
   ```
   d_loss_real_pair = binary_cross_entropy_with_logits_loss(
       d_score_on_real_pair, real_label_tensor,    # all 0.9 (label smoothing)
   )
   d_loss_fake_pair = binary_cross_entropy_with_logits_loss(
       d_score_on_fake_pair, fake_label_tensor,    # all 0.0
   )
   total_d_loss = d_loss_real_pair + d_loss_fake_pair
   ```
4. `total_d_loss.backward()` → gradient flows through D only.
   `discriminator_optimizer.step()` updates D's weights.

### G-step (updates G **and** E, R)

1. Rebuild `clean_triple_embedding_for_g_step` **with gradient** — this is
   the path through which E and R will learn.
2. Sample TWO latents `latent_sample_1`, `latent_sample_2` (needed for the
   diversity regulariser).
3. Forward G twice:
   `generator_output_z1 = generator_model(clean, latent_sample_1)`,
   `generator_output_z2 = generator_model(clean, latent_sample_2)`.
4. Re-embed both candidates: `candidate_embedding_from_z1`,
   `candidate_embedding_from_z2`.
5. Compute the three losses:
   ```
   loss_reconstruction = (
       cross_entropy_loss(generator_output_z1['head_entity_logits'], target_triple_indices_batch[:, 0])
     + cross_entropy_loss(generator_output_z1['relation_logits'],    target_triple_indices_batch[:, 1])
     + cross_entropy_loss(generator_output_z1['tail_entity_logits'], target_triple_indices_batch[:, 2])
   )
   loss_adversarial = binary_cross_entropy_with_logits_loss(
       d_score_on_fake_pair_for_g_step, real_label_tensor,
   )
   loss_diversity = mode_seeking_diversity_loss(
       candidate_embedding_from_z1, candidate_embedding_from_z2,
       latent_sample_1, latent_sample_2,
   )
   total_g_loss = (
       loss_weight_reconstruction * loss_reconstruction
     + loss_weight_adversarial    * loss_adversarial
     + loss_weight_diversity      * loss_diversity
   )
   ```
6. `total_g_loss.backward()` → gradient flows through G,
   `re_embed_soft_samples`, the Gumbel-Softmax, the bottleneck, the
   encoder, AND `clean_triple_embedding_for_g_step` (back into E and R).
   `generator_optimizer.step()` updates G, E, R all at once.

### What each loss term does, in plain language

| Term | Weight | What it teaches |
|---|---|---|
| `loss_reconstruction` (CE on three heads) | 20 | "Given this clean triple, the right corrupted output is `(target_h, target_r, target_t)` — match it." This is the dominant signal. |
| `loss_adversarial` (cGAN BCE) | 1 | "Make D believe your output is real." Sharpening signal; pushes outputs onto the manifold of plausible corruptions. |
| `loss_diversity` (mode-seeking) | 0.5 | "Different latents `latent_sample_1`, `latent_sample_2` should produce different outputs." Defends against posterior collapse where G ignores `z`. |

**Why CE for reconstruction instead of L1?** Pix2Pix originally uses L1
because pixels are continuous. Our output is categorical (which entity /
which relation). Cross-entropy is the categorical analogue of L1 — same
role, mathematically appropriate type.

## 9. Stage 7 — Inference (clean KG → K diverse corrupted KGs)

**Call site (notebook Step 10):**

```python
corrupted_kgs = generate_k_corrupted_kgs(
    generator_model, entity_embedding, relation_embedding,
    clean_triples_in_kg,
    K_samples=num_inference_samples,                  # 20
    num_corruption_steps=num_corruption_steps_per_round,
    latent_dim=latent_dim,
    gumbel_temperature=0.5,
    device=compute_device,
    random_generator=np.random.default_rng(7777),
)
```

**Code:** [inference/two_stage.py](inference/two_stage.py).

Two-stage process per sample:

```
for sample_index in range(K_samples=20):
    # Stage A: random selection (rule-based, mirrors TRIC)
    selected_triple_indices = rng.choice(18, size=num_corruption_steps=2, replace=False)

    # Stage B: per-triple G forward
    corrupted_kg = []
    for position_index, (h, r, t) in enumerate(clean_triples_in_kg):
        if position_index not in selected_triple_indices:
            corrupted_kg.append((h, r, t))   # pass-through
            continue
        latent_sample = torch.randn(1, latent_dim=8)
        generator_output = generator_model(
            triple_emb(h, r, t), latent_sample,
            gumbel_temperature=0.5, return_soft_samples=False,
        )
        corrupted_kg.append((
            argmax(generator_output['head_entity_logits']),
            argmax(generator_output['relation_logits']),
            argmax(generator_output['tail_entity_logits']),
        ))
    all_corrupted_kgs.append(corrupted_kg)
```

For `K_samples=20`: 20 corrupted versions of the same clean KG. Diversity
has two sources:

- **Stage A randomness** — each sample picks 2 *different* triples to corrupt.
- **Stage B randomness** — even on the same triple, different
  `latent_sample` + Gumbel-Softmax can yield different outputs.

## 10. Worked example — trace one training pair from disk to gradient

Pick the first training pair in the tensor: `triple_pair_tensor[0]` =
`[0, 0, 1, 0, 1, 1]`.

Resolved via `node_labels` and `relation_names`:

- Clean: `(0, 0, 1)` = `(Alice, born_in, Australia)`
- Target: `(0, 1, 1)` = `(Alice, married_to, Australia)`

A pure `change_relation` edit. Following it through the pipeline:

**(a) Embedding lookup**

```
head_embedding     = entity_embedding(0)    →  100-dim "Alice" vector
relation_embedding_vector = relation_embedding(0)  →  100-dim "born_in" vector
tail_embedding     = entity_embedding(1)    →  100-dim "Australia" vector
clean_triple_embedding_for_g_step = stack([head_embedding, relation_embedding_vector, tail_embedding])
                                  →  shape (3, 100)
```

**(b) Generator forward pass**

- Sample `latent_sample_1 = randn(8) ≈ [0.7, -0.3, 0.2, …]`
- Encoder → bottleneck → `bottleneck_hidden_state` (128-dim)
- Three heads produce logits:
  - `head_entity_logits[0]` should be highest (keep Alice as head)
  - `relation_logits[1]` should be highest (substitute married_to for
    born_in)
  - `tail_entity_logits[1]` should be highest (keep Australia as tail)
- Gumbel-Softmax samples near-one-hot from each
- Re-embed → `candidate_embedding_from_z1 ≈ stack(E[0], R[1], E[1])` of
  shape (3, 100)

**(c) Reconstruction loss**

```
loss_reconstruction_head = cross_entropy_loss(head_entity_logits, target=0)
                           → penalises if argmax ≠ 0
loss_reconstruction_relation = cross_entropy_loss(relation_logits, target=1)
                               → penalises if argmax ≠ 1
loss_reconstruction_tail = cross_entropy_loss(tail_entity_logits, target=1)
                           → penalises if argmax ≠ 1
loss_reconstruction = loss_reconstruction_head + loss_reconstruction_relation + loss_reconstruction_tail
```

**(d) Adversarial loss**

- D scores the pair
  `(clean_triple_embedding_for_g_step, candidate_embedding_from_z1)`.
- BCE pushes the score toward "real".

**(e) Diversity loss**

- A second forward pass with `latent_sample_2` produces
  `candidate_embedding_from_z2`.
- The two candidates should differ (model uses `z`).

**(f) Backward**

- `total_g_loss.backward()` updates G, plus `E[0]`, `E[1]`, `R[0]`, `R[1]`
  (every embedding row that was indexed in this batch).

Repeat ~7,640 / 64 ≈ **119 batches per epoch × 400 epochs ≈ 47,600 such
gradient updates** to fully train the model.

## 11. Data dictionary (which variable is what?)

| Variable | Where defined | Type | Shape | What it is |
|---|---|---|---|---|
| `knowledge_graph` | Step 2 | `KnowledgeGraph` | — | The loaded KG namespace |
| `knowledge_graph.triples_idx` | Step 2 | `List[(int,int,int)]` | 18 triples | Clean KG in vocab-indexed form |
| `clean_triples_in_kg` | Step 2 | `List[(int,int,int)]` | 18 triples | Alias of the above for inference and the cells below |
| `num_entities` | Step 2 | `int` | scalar = 10 | Entity vocab size \|E\| |
| `num_relations` | Step 2 | `int` | scalar = 3 | Relation vocab size \|R\| |
| `tric_corruption_fn` | Step 4 | `Callable` | — | TRIC with kwargs pre-bound via `functools.partial` |
| `triple_pair_tensor` | Step 4 | `torch.long` | `(M, 6)` | All `(clean, target)` training pairs |
| `training_clean_triples` | Step 4 | `torch.long` | `(M, 3)` | First three columns |
| `training_target_triples` | Step 4 | `torch.long` | `(M, 3)` | Last three columns |
| `triple_pair_dataloader` | Step 4 | `DataLoader` | batches of 64 | Yields `(clean_triple_indices_batch, target_triple_indices_batch)` |
| `entity_embedding` | Step 6 | `EntityEmbedding` | `E ∈ (10, 100)` | `nn.Embedding` lookup table |
| `relation_embedding` | Step 6 | `RelationEmbedding` | `R ∈ (3, 100)` | `nn.Embedding` lookup table |
| `generator_model` | Step 6 | `TripleGenerator` | ~1.8 M params | The G network |
| `discriminator_model` | Step 6 | `TripleDiscriminator` | ~187 k params | The D network |
| `clean_triple_indices_batch` | Step 8 (in loop) | `torch.long` | `(64, 3)` | Per-batch clean index tensor |
| `target_triple_indices_batch` | Step 8 (in loop) | `torch.long` | `(64, 3)` | Per-batch target index tensor |
| `clean_triple_embedding_for_g_step` | Step 8 | `torch.float` | `(64, 3, 100)` | Embeddings of the clean triples (G-step) |
| `target_triple_embedding_for_d_step` | Step 8 | `torch.float` | `(64, 3, 100)` | Embeddings of the targets (D-step) |
| `latent_sample_1`, `latent_sample_2` | Step 8 | `torch.float` | `(64, 8)` | Per-step latent samples for the mode-seeking reg |
| `generator_output_z1['head_entity_logits']` etc. | Step 8 | `torch.float` | `(64, 10)`, `(64, 3)`, `(64, 10)` | Generator output logits |
| `candidate_embedding_from_z1`, `candidate_embedding_from_z2` | Step 8 | `torch.float` | `(64, 3, 100)` | Two candidate triple embeddings (different latents) |
| `corrupted_kgs` | Step 10 | `List[List[(int,int,int)]]` | `20 × 18` | K corrupted KGs from inference |

## 12. Code-location reference

| Concept | File | Function/class |
|---|---|---|
| TSV → namespace | `kg_data/loader.py` | `load_kg`, `KnowledgeGraph` |
| Triples → adjacency (visualisation only) | `kg_data/adjacency.py` | `triples_to_adjacency_tensor` |
| Triple-level TRIC corruption | `corruption_strategies/tric.py` | `apply_tric_corruption` |
| `(M, 6)` training-pair tensor | `dataset_builders/triple_pair_dataset.py` | `build_triple_pair_dataset` |
| Entity/relation embeddings | notebook Step 6 | `EntityEmbedding`, `RelationEmbedding` |
| Gumbel-Softmax | notebook Step 6 | `gumbel_softmax_sample` |
| Generator | notebook Step 6 | `TripleGenerator` |
| Discriminator | notebook Step 6 | `TripleDiscriminator` |
| Re-embed soft samples | notebook Step 6 | `re_embed_soft_samples` |
| Alternating G/D training | notebook Step 8 | (inline in cell) |
| Mode-seeking diversity reg | notebook Step 8 | `mode_seeking_diversity_loss` |
| Two-stage inference | `inference/two_stage.py` | `generate_k_corrupted_kgs` |
| Operation-mix categorisation | `evaluation/operation_mix.py` | `categorise_corruption`, `summarise_operation_mix` |
| Distribution validation | `evaluation/distribution_match.py` | `validate_distribution` |

## 13. Recommended reading order

1. Read this whole document end to end.
2. Read the **PyTorch primer** markdown cell at the top of the notebook
   if you're new to PyTorch.
3. Open the notebook and step through cells. For each cell, locate the
   matching Stage in this doc.
4. After Step 4 runs, inspect `triple_pair_tensor[0:5]` and translate the
   indices to entity / relation names using `node_labels` and
   `relation_names`.
5. Read `TripleGenerator` in Step 6 line by line. Match each layer to
   Stage 4 here.
6. After training (Step 8), print
   `entity_embedding.embedding_table.weight.shape` and
   `entity_embedding.embedding_table.weight[0][:5]` — confirm the embedding
   has learned (values should not be the initial random ones).
7. After inference (Step 10), inspect `corrupted_kgs[0]`. Translate the
   indices to names. Identify which 2 triples got corrupted; confirm
   they're `change_relation` style.

## 14. Self-test questions

If you can answer these, you understand the pipeline:

1. The loader produces *two* forms of the triples (`triples` strings,
   `triples_idx` ints). Which form does the model consume, and why does
   the other form exist?
2. What is `knowledge_graph.triple_set_idx` for, and why is it a `set`
   rather than a `list`?
3. In Stage 2, why do we permute entity indices on each round before
   applying TRIC?
4. What is the shape of `clean_triple_embedding_for_g_step` for
   `batch_size = 64`?
5. The generator emits *three* softmaxes (head_entity, relation,
   tail_entity). Which one makes `change_relation` corruption possible?
   Why doesn't Dai et al.'s single softmax suffice?
6. In the D-step, why are embeddings built under `torch.no_grad()`?
7. The Gumbel-Softmax output is "near-one-hot but differentiable". What
   would break if we used plain `argmax` instead?
8. After training, can two independent inference runs with the same clean
   KG produce *different* corrupted outputs? If yes, what are the two
   sources of randomness?
9. The pipeline never builds an adjacency tensor. Where does adjacency get
   built on-demand, and why only there?
10. If we replaced `data/dummy_kg` with FB15k-237 (N=14.5k, R=237),
    which parts of the pipeline would scale unchanged, and which would
    slow down? Why is the bottleneck not memory anymore?
