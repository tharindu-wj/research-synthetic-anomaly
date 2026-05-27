# research_synthetic_anomaly

A GAN that emits synthetic anomalous triples for the ADKGD knowledge-graph
anomaly detector (Wu et al., 2025; arXiv:2501.07078). Output is a TSV of
~400 k unique non-real triples consumed by ADKGD's
`Reader.load_gan_negatives(...)` hook with `--neg_source gan`. Replaces
ADKGD's default random corruption with type-coherent-but-wrong negatives
learned from the graph.

The model is a Pix2Pix-style conditional GAN trained on TRIC corruption
pairs (Senaratne ESWC 2023): per clean triple, the generator emits a
substitution at one of `{h, r, t}`, with the entity / relation
embeddings learned jointly. Sampling masks the clean index at the
chosen position so every emitted triple differs from its input.

## Folder structure

```
.
├── data/                       Datasets (input)
│   └── dummy_kg/               10 entities, 3 relations, 18 triples - smoke-test KG
│       ├── train.txt           tab-separated head, relation, tail
│       ├── valid.txt           validation split (may be empty)
│       ├── test.txt            test split (may be empty)
│       ├── entity_metadata.txt entity_id <TAB> display_name <TAB> type
│       └── relation_metadata.txt
│
│   (data/FB15K/ is git-ignored; drop train.txt / valid.txt / test.txt
│    here when you're ready to train on FB15k-237.)
│
├── src/                        Live ADKGD pipeline source (importable as a package root)
│   ├── kg_data/
│   │   ├── loader.py                       load_kg() and load_kg_union()
│   │   └── relation_signature_types.py    derive entity pseudo-types from
│   │                                      per-entity (relation, position)
│   │                                      bags via TF-IDF + SVD + KMeans
│   ├── corruption_strategies/
│   │   └── tric.py                         seven-operation TRIC corruption
│   ├── dataset_builders/
│   │   └── triple_pair_dataset.py          (clean, corrupted) training pairs
│   ├── models/
│   │   └── triple_gan.py                   EntityEmbedding, RelationEmbedding,
│   │                                       TripleGenerator, TripleDiscriminator,
│   │                                       gumbel_softmax_sample
│   ├── training/
│   │   └── train_triple_gan.py             alternating G/D loop, hybrid loss
│   ├── sampling/
│   │   ├── sample_anomalies.py             single-position masked Gumbel decode
│   │   └── tsv_writer.py                   indices -> string IDs, LF endings
│   ├── baseline_random/
│   │   └── random_anomaly_generator.py     Las-Vegas random TSV (control)
│   └── validation/
│       └── preflight.py                    pre-flight validator
│
├── scripts/                    CLI entry points (one per pipeline stage)
│   ├── build_pseudo_types.py
│   ├── build_random_baseline_tsv.py
│   ├── train_gan.py
│   ├── generate_gan_tsv.py
│   └── validate_tsv.py
│
├── outputs/                    Generated artifacts (git-ignored)
│   ├── checkpoints/            *.pt files saved by train_gan.py
│   ├── gan_negatives.tsv       final deliverable for ADKGD
│   └── random_baseline.tsv     control TSV
│
├── slurm/
│   └── run_gan_train.slurm    GPU SLURM wrapper: train_gan.py + generate_gan_tsv.py
│                              on Flinders DeepThought (Tesla V100)
│
├── legacy_notebook/            Frozen research notebook + its exclusive deps
│   ├── knowledge_graph_clone.ipynb
│   ├── inference/              generate_k_corrupted_kgs (notebook-only)
│   ├── evaluation/             operation_mix, distribution_match (notebook-only)
│   ├── visualisation/          NetworkX KG plots
│   ├── legacy_helpers/         triples_to_adjacency_tensor, write_dummy_kg
│   ├── run_notebook.py         headless executor
│   ├── slurm/                  notebook-specific SLURM scripts
│   ├── pipeline_explainer.md   notebook data-flow docs
│   └── README.md               read this if touching the notebook
│
├── baseline/                   ADKGD reference paper + source
└── requirements.txt
```

`data/`, `src/`, `scripts/`, `outputs/`: **live ADKGD pipeline.**
`legacy_notebook/`: **frozen research snapshot** — do not edit; all
model / training changes happen in `src/models/` and `src/training/`.

## Setup

```bash
pip install -r requirements.txt
# torch is installed separately so you can pick the CPU or CUDA wheel:
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.1
```

The scripts add `src/` to `sys.path` themselves — no install step
needed, just run them from the repo root.

HPC setup for the live ADKGD pipeline (Phase 5 SLURM wrapper) is
forthcoming. Setup notes for the legacy notebook on Flinders DeepThought
live at
[legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md](legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md).

## End-to-end usage

The full path from raw KG TSVs to an ADKGD-ready negatives file:

```bash
# 1. Derive entity pseudo-types from the relation signature.
#    Writes <data>/entity_metadata.txt that the loader picks up
#    next time. Required for TRIC's swap_*_same_type operations to fire.
python scripts/build_pseudo_types.py --data data/FB15K

# 2. Train the GAN. Saves outputs/checkpoints/fb15k.pt.
python scripts/train_gan.py \
    --data data/FB15K \
    --out  outputs/checkpoints/fb15k.pt \
    --epochs 400 --batch-size 256 --device cuda

# 3. Sample 400k unique non-real triples and write the TSV.
#    Automatically runs preflight at the end; aborts on failure.
python scripts/generate_gan_tsv.py \
    --data data/FB15K \
    --ckpt outputs/checkpoints/fb15k.pt \
    --target 400000 \
    --out   outputs/gan_negatives.tsv \
    --device cuda

# 4. (optional) Re-run preflight standalone.
python scripts/validate_tsv.py outputs/gan_negatives.tsv \
    --data data/FB15K --target 400000

# 5. Ship to ADKGD:
scp outputs/gan_negatives.tsv  <hpc>:adkgd/data/FB15K/gan_negatives.tsv
# Then on the ADKGD side: --neg_source gan
```

### Phase 5: HPC run (Flinders DeepThought)

Steps 2 and 3 above are wrapped into one GPU SLURM job at
[slurm/run_gan_train.slurm](slurm/run_gan_train.slurm). It allocates one
Tesla V100, asserts `torch.cuda.is_available()`, verifies
`data/FB15K/entity_metadata.txt` exists (run `build_pseudo_types.py` on
the login node first), runs `train_gan.py` then `generate_gan_tsv.py`,
and copies the final TSV into `~/scratch/adkgd_runs/<jobid>/`.

```bash
# One-time login-node setup (see legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md
# for the full env recipe).
python scripts/build_pseudo_types.py --data data/FB15K

# Submit (defaults: 400 epochs, batch 256, target 400k, 4h time, 16G mem):
sbatch slurm/run_gan_train.slurm

# Override hyperparameters via env vars - no script edit needed:
EPOCHS=200 BATCH_SIZE=512 sbatch slurm/run_gan_train.slurm
DATASET_DIR=data/other_kg TARGET_POOL=200000 sbatch slurm/run_gan_train.slurm

# Track and collect:
squeue -u $USER
tail -f adkgd_gan-<jobid>.out.txt
ls ~/scratch/adkgd_runs/<jobid>/
```

The job ends `COMPLETED` and `gan_negatives.tsv` passes preflight when
all four checks (volume, uniqueness, vocab coverage, real-graph
collisions) come up clean. Full HPC user guide:
[docs/deepthoughtdocs-flinders-edu-au-en-latest.pdf](docs/deepthoughtdocs-flinders-edu-au-en-latest.pdf).

### Random-baseline control

Mirrors ADKGD's own random corruption — useful for rehearsing the
cluster integration before the GAN is trained, and as an honest
comparison point for the final metric:

```bash
python scripts/build_random_baseline_tsv.py \
    --data data/FB15K --target 400000 \
    --out  outputs/random_baseline.tsv
```

## Scripts reference

### `build_pseudo_types.py`

Cluster entities by their (relation, position) participation
signature → pseudo-types written to
`<data_dir>/entity_metadata.txt` in the 3-column format the loader
already reads. Without this file, TRIC silently disables its
`swap_subject_same_type` / `swap_object_same_type` operations and
the model loses its type-coherence training signal.

```bash
python scripts/build_pseudo_types.py --data data/FB15K
python scripts/build_pseudo_types.py --data data/dummy_kg --clusters 3 --svd-dim 4 \
    --out outputs/dummy_pseudo_types.txt   # use --out to avoid clobbering
                                            # the dummy KG's hand-curated types
```

Flags:
- `--clusters N`    target number of pseudo-types (default 100; the
                    helper clamps to `num_entities`).
- `--svd-dim D`     truncated-SVD components before clustering (default 32).
- `--seed S`        random seed for SVD + KMeans.
- `--out PATH`      destination (defaults to `<data>/entity_metadata.txt`).

### `build_random_baseline_tsv.py`

Las-Vegas random sampler: pick uniform `(h, r, t)`, retry on
self-loops / real-graph collisions / duplicates until pool ≥ target.
Mirrors ADKGD's own `generate_anomalous_triples_2`. Use as a control
in the metric comparison, or to rehearse the ADKGD integration before
the real GAN is trained.

```bash
python scripts/build_random_baseline_tsv.py \
    --data data/FB15K --target 400000 \
    --out  outputs/random_baseline.tsv --seed 0
```

### `train_gan.py`

Train the `TripleGenerator + TripleDiscriminator` cGAN. Loads the
KG via `load_kg_union`, builds TRIC corruption pairs, runs the
alternating loop with the hybrid loss `20·L_recon + 1·L_adv + 0.5·L_div`,
checkpoints to `--out`.

```bash
python scripts/train_gan.py --data data/dummy_kg \
    --epochs 50 --batch-size 32 --device cpu \
    --out outputs/checkpoints/dummy.pt

python scripts/train_gan.py --data data/FB15K \
    --epochs 400 --batch-size 256 --device cuda \
    --out outputs/checkpoints/fb15k.pt
```

Key flags:
- `--epochs`, `--batch-size`, `--device {cuda,cpu}`, `--lr`
- `--embedding-dim` (200), `--latent-dim` (32), `--bottleneck-hidden-dim` (512)
- `--training-rounds` (4000), `--corruption-steps-per-round` (2)
- `--w-recon`, `--w-adv`, `--w-div` (loss weights; defaults 20 / 1 / 0.5)
- `--checkpoint-every N` save intermediate checkpoints every N epochs.
- `--restrict-tric-to-change-relation` notebook's dummy-KG setting; do
  not use for FB15k-237 where the full TRIC mix is what we want.

### `generate_gan_tsv.py`

Load a checkpoint, sample a deduplicated pool of synthetic anomalies,
write the TSV, run preflight. The default decoding strategy
(`masked_single_position=True`) per sample randomly picks ONE of
`{h, r, t}`, sets the clean index's logit to `-inf`, adds Gumbel
noise, then argmax. Other two positions copy the input. This mirrors
TRIC's substitution semantics and prevents the model collapsing to
identity output.

```bash
python scripts/generate_gan_tsv.py \
    --data data/FB15K --ckpt outputs/checkpoints/fb15k.pt \
    --target 400000 --out outputs/gan_negatives.tsv --device cuda \
    --samples-per-triple 2 --max-passes 5 --batch-size 256
```

Flags:
- `--target N`             pool size to fill (e.g. 400000 for FB15k-237).
- `--samples-per-triple K` Gumbel samples per real triple per pass (default 2).
- `--max-passes`           number of full passes over the real triples (default 5).
- `--batch-size`           forward-pass batch size.
- `--gumbel-temperature`   default 0.5; lower is sharper.
- `--no-masked-single-position` disable the masked decode (diagnostic only;
                                 expect near-zero non-collision yield).
- `--skip-preflight`        skip the auto-run preflight at the end.

### `validate_tsv.py`

Stand-alone preflight: line count, uniqueness, ADKGD vocab match,
real-graph collision count. Exits non-zero on any failure — chainable
into shell pipelines.

```bash
python scripts/validate_tsv.py outputs/gan_negatives.tsv \
    --data data/FB15K --target 400000
```

Flags:
- `--target`, `--min-unique-ratio` (0.95), `--max-unknown-vocab-rate` (0.05),
  `--max-collision-rate` (0.05).

## Smoke test (dummy KG, CPU, ~30 s)

Quick verification that the whole pipeline works after a fresh checkout:

```bash
python scripts/build_random_baseline_tsv.py --data data/dummy_kg --target 50 \
    --out outputs/dummy_random.tsv
python scripts/validate_tsv.py outputs/dummy_random.tsv \
    --data data/dummy_kg --target 50

python scripts/train_gan.py --data data/dummy_kg \
    --epochs 30 --batch-size 16 --device cpu \
    --embedding-dim 64 --latent-dim 8 --bottleneck-hidden-dim 128 \
    --training-rounds 1000 \
    --out outputs/checkpoints/dummy.pt

python scripts/generate_gan_tsv.py --data data/dummy_kg \
    --ckpt outputs/checkpoints/dummy.pt \
    --target 50 --out outputs/dummy_gan.tsv --device cpu \
    --samples-per-triple 8 --max-passes 20 --batch-size 16
```

All five commands must exit 0; both TSVs must end with
`Verdict: OK to use.`.

## The legacy notebook

`legacy_notebook/knowledge_graph_clone.ipynb` is the original research
notebook that produced the GAN architecture and training loop now
extracted into `src/models/triple_gan.py` and
`src/training/train_triple_gan.py`. **Frozen**: it remains runnable
end-to-end but is no longer part of the ADKGD pipeline.

To re-execute locally:

```bash
python legacy_notebook/run_notebook.py legacy_notebook/knowledge_graph_clone.ipynb
```

For HPC (Flinders DeepThought, SLURM):
[legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md](legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md).
See [legacy_notebook/README.md](legacy_notebook/README.md) for what's
in that directory.
