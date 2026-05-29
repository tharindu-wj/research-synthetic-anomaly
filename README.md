# research_synthetic_anomaly

A GAN that emits a **per-positive hashmap** of synthetic anomalous
triples for the ADKGD knowledge-graph anomaly detector (Wu et al., 2025;
arXiv:2501.07078). For each unique real triple `(h, r, t)` in the
dataset, the hashmap stores **one** GAN-picked corruption — a triple
that differs from the input at exactly one of the three positions
(head, rel, or tail). The slot to corrupt is picked uniformly at
random per positive at export time (mirroring ADKGD's existing 1/3
slot distribution).

ADKGD's training-time negative sampler (`Reader.get_data`) just builds
the `(h_str, r_str, t_str)` 3-tuple from a positive and looks up the
corruption directly — JavaScript-object-style:

```python
table[(orig_h, orig_r, orig_t)] = (neg_h, neg_r, neg_t)
neg = table.get((h_str, r_str, t_str))
```

The model is a Pix2Pix-style conditional GAN trained on TRIC corruption
pairs (Senaratne ESWC 2023). The integration is **decoupled by file**:
kggan and ADKGD live in separate envs and only communicate through the
hashmap TSV. No Python import boundary, no shared GPU.

## Folder structure

```
.
├── data/                       Datasets (input)
│   └── dummy_kg/               toy KG for smoke tests
│       ├── train.txt           tab-separated head, relation, tail
│       ├── valid.txt           validation split (may overlap with train)
│       ├── test.txt            test split
│       ├── entity_metadata.txt entity_id <TAB> display_name <TAB> type
│       └── relation_metadata.txt
│
│   (data/FB15K/ is git-ignored; drop train.txt / valid.txt / test.txt
│    here when you're ready to train on FB15k-237.)
│
├── src/                        kggan source (importable as a package root)
│   ├── kg_data/
│   │   ├── loader.py                       load_kg(), load_kg_union()
│   │   └── relation_signature_types.py    derive entity pseudo-types from
│   │                                      per-entity (relation, position)
│   │                                      bags via TF-IDF + SVD + KMeans
│   ├── corruption_strategies/
│   │   └── tric.py                         seven-operation TRIC corruption
│   ├── dataset_builders/
│   │   └── triple_pair_dataset.py          (clean, corrupted) training pairs
│   ├── models/
│   │   └── triple_gan.py                   EntityEmbedding, RelationEmbedding,
│   │                                       TripleGenerator, TripleDiscriminator
│   ├── training/
│   │   └── train_triple_gan.py             alternating G/D loop, hybrid loss
│   ├── sampling/
│   │   └── hashmap_export.py               6-column hashmap exporter + decode
│   │                                       helpers + checkpoint loader
│   └── validation/
│       └── preflight_hashmap.py            6-column TSV validator
│
├── scripts/                    CLI entry points
│   ├── build_pseudo_types.py
│   ├── train_gan.py
│   ├── generate_gan_hashmap.py    export the hashmap TSV consumed by ADKGD
│   └── validate_hashmap.py        stand-alone preflight check
│
├── outputs/                    Generated artefacts (git-ignored)
│   ├── checkpoints/            *.pt saved by train_gan.py
│   └── gan_hashmap.tsv         final deliverable for ADKGD
│
├── slurm/
│   └── run_gan_train.slurm     GPU SLURM wrapper (Tesla V100 on DeepThought):
│                               trains the GAN then exports the hashmap
│
├── legacy_notebook/            Frozen research notebook + its exclusive deps
│   ├── knowledge_graph_clone.ipynb
│   ├── inference/, evaluation/, visualisation/, legacy_helpers/
│   ├── run_notebook.py, slurm/, pipeline_explainer.md
│   └── README.md
│
├── baseline/                   ADKGD reference paper + source
└── requirements.txt
```

`data/`, `src/`, `scripts/`, `outputs/`: **live ADKGD pipeline.**
`legacy_notebook/`: **frozen research snapshot** — do not edit; all
model and training changes happen in `src/models/` and
`src/training/`.

## Setup

```bash
pip install -r requirements.txt
# torch is installed separately so you can pick the CPU or CUDA wheel:
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121 # CUDA 12.1
```

The scripts add `src/` to `sys.path` themselves — no install step
needed, just run them from the repo root.

HPC setup notes for the legacy notebook on Flinders DeepThought:
[legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md](legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md)
covers the conda env that the live pipeline also uses.

## End-to-end usage

From raw KG TSVs to an ADKGD-ready hashmap:

```bash
# 1. Derive entity pseudo-types from the relation signature.
#    Writes <data>/entity_metadata.txt that the loader picks up.
#    Required for TRIC's swap_*_same_type operations to fire during
#    training. Skip if your dataset already has entity_metadata.txt.
python scripts/build_pseudo_types.py --data data/FB15K

# 2. Train the GAN. Saves outputs/checkpoints/fb15k.pt.
python scripts/train_gan.py \
    --data data/FB15K \
    --out  outputs/checkpoints/fb15k.pt \
    --epochs 400 --batch-size 256 --device cuda

# 3. Export the hashmap TSV. Auto-runs preflight at the end and aborts
#    if any check fails. One row per unique real triple - slot picked
#    uniformly at random per positive at export time.
python scripts/generate_gan_hashmap.py \
    --data data/FB15K \
    --ckpt outputs/checkpoints/fb15k.pt \
    --out  outputs/gan_hashmap.tsv \
    --device cuda

# 4. (optional) Re-run preflight standalone.
python scripts/validate_hashmap.py outputs/gan_hashmap.tsv \
    --data data/FB15K

# 5. Ship to ADKGD:
scp outputs/gan_hashmap.tsv  <hpc>:adkgd/data/FB15K/gan_hashmap.tsv
# Then on the ADKGD side: --neg_source gan_hashmap --gan_hashmap_path data/FB15K/gan_hashmap.tsv
```

### Phase 5: HPC run (Flinders DeepThought)

Steps 2 and 3 are wrapped into one GPU SLURM job at
[slurm/run_gan_train.slurm](slurm/run_gan_train.slurm) (job name:
`kggan_train`). It allocates one Tesla V100, asserts
`torch.cuda.is_available()`, verifies `data/FB15K/entity_metadata.txt`
exists, runs `train_gan.py` then `generate_gan_hashmap.py`, and copies
the final hashmap into `~/scratch/kggan_runs/<jobid>/`.

```bash
# One-time login-node setup (see legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md
# for the full env recipe).
python scripts/build_pseudo_types.py --data data/FB15K

# Submit (defaults: 400 epochs, batch 256, 4h time, 16G mem):
sbatch slurm/run_gan_train.slurm

# Override hyperparameters via env vars - no script edit needed:
EPOCHS=200 BATCH_SIZE=512 sbatch slurm/run_gan_train.slurm
DATASET_DIR=data/other_kg sbatch slurm/run_gan_train.slurm

# Track and collect:
squeue -u $USER
tail -f kggan_train-<jobid>.out.txt
ls ~/scratch/kggan_runs/<jobid>/
```

The job ends `COMPLETED` and the hashmap passes preflight when all
structural and content checks come up clean. Full HPC user guide:
[docs/deepthoughtdocs-flinders-edu-au-en-latest.pdf](docs/deepthoughtdocs-flinders-edu-au-en-latest.pdf).

## ADKGD-side integration

ADKGD's `Reader.get_data()` already has a dispatch on `args.neg_source`
(the existing `gan` branch loads a pool-style TSV). Add a third branch
for `gan_hashmap`:

```python
# In Reader.get_data():
neg_source = getattr(self.args, 'neg_source', 'random')
if neg_source == 'gan':
    bn_triples = self.load_gan_negatives(self.args.gan_neg_path, n=len(bp_triples))
elif neg_source == 'gan_hashmap':
    bn_triples = self._gan_hashmap_negatives(bp_triples)
else:
    bn_triples = self.generate_anomalous_triples(bp_triples)

# New helper - 3-tuple key, no slot logic:
def _gan_hashmap_negatives(self, pos_triples):
    if not hasattr(self, '_gan_hashmap') or self._gan_hashmap is None:
        self._gan_hashmap = self._load_gan_hashmap(self.args.gan_hashmap_path)
    out, hits, fallbacks = [], 0, 0
    for h, r, t in pos_triples:
        key = (self.id2ent[h], self.id2rel[r], self.id2ent[t])
        neg_str = self._gan_hashmap.get(key)
        if neg_str is not None:
            try:
                out.append((self.ent2id[neg_str[0]],
                            self.rel2id[neg_str[1]],
                            self.ent2id[neg_str[2]]))
                hits += 1
                continue
            except KeyError:
                pass  # unknown vocab in the value -> fall through to fallback
        fallbacks += 1
        out.extend(self.generate_anomalous_triples([(h, r, t)]))
    print('[GAN hashmap] hits: %d / %d, fallbacks: %d'
          % (hits, len(pos_triples), fallbacks))
    return out

def _load_gan_hashmap(self, path):
    table = {}
    with open(path, encoding='utf-8') as f:
        for raw in f:
            parts = raw.rstrip('\n').split('\t')
            if len(parts) != 6:
                continue
            orig_h, orig_r, orig_t, neg_h, neg_r, neg_t = parts
            table[(orig_h, orig_r, orig_t)] = (neg_h, neg_r, neg_t)
    print('[GAN hashmap] loaded %d entries from %s' % (len(table), path))
    return table
```

New CLI args on the ADKGD side:
- `--neg_source gan_hashmap`
- `--gan_hashmap_path outputs/gan_hashmap.tsv`

Everything else in ADKGD (positives, loss, eval) is untouched.

## Hashmap TSV format

Six tab-separated columns, **one row per unique real triple**:

```
orig_h <TAB> orig_r <TAB> orig_t <TAB> neg_h <TAB> neg_r <TAB> neg_t
```

- The negative differs from the original at exactly ONE of the three
  positions (head, rel, or tail). The validator enforces this. Which
  position moves is picked uniformly at random per positive at export
  time and NOT stored on disk — ADKGD doesn't need it.
- UTF-8, LF endings, no header, sorted by `(orig_h, orig_r, orig_t)`
  so diff-stable across runs with the same seed.
- Total rows = `len(unique union triples)`. For FB15K-237 union
  (~310k unique triples) that's ~310k rows, ~47 MB on disk.

ADKGD keys the dict by the `(orig_h, orig_r, orig_t)` 3-tuple of
strings (Python tuples hash natively — no composite-string key
needed). Real triples not in the hashmap fall back to ADKGD's own
random corruptor.

## Scripts reference

### `build_pseudo_types.py`

Cluster entities by their (relation, position) participation signature
→ pseudo-types written to `<data_dir>/entity_metadata.txt`. Without
this file, TRIC silently disables its `swap_subject_same_type` /
`swap_object_same_type` operations and the model loses its
type-coherence training signal.

```bash
python scripts/build_pseudo_types.py --data data/FB15K
python scripts/build_pseudo_types.py --data data/dummy_kg --clusters 3 --svd-dim 4 \
    --out outputs/dummy_pseudo_types.txt   # use --out to avoid clobbering
                                            # the dummy KG's hand-curated types
```

### `train_gan.py`

Train the `TripleGenerator + TripleDiscriminator` cGAN. Loads the KG
via `load_kg_union`, builds TRIC corruption pairs, runs the alternating
loop with the hybrid loss `20·L_recon + 1·L_adv + 0.5·L_div`,
checkpoints to `--out`.

```bash
python scripts/train_gan.py --data data/dummy_kg \
    --epochs 50 --batch-size 32 --device cpu \
    --out outputs/checkpoints/dummy.pt

python scripts/train_gan.py --data data/FB15K \
    --epochs 400 --batch-size 256 --device cuda \
    --out outputs/checkpoints/fb15k.pt
```

Key flags: `--epochs`, `--batch-size`, `--device {cuda,cpu}`, `--lr`,
`--embedding-dim` (200), `--latent-dim` (32),
`--bottleneck-hidden-dim` (512), `--training-rounds` (4000),
`--w-recon` / `--w-adv` / `--w-div` (loss weights; defaults 20 / 1 /
0.5), `--checkpoint-every N`.

### `generate_gan_hashmap.py`

Load a checkpoint, walk each unique real triple in the union graph,
pick one slot uniformly at random per positive, and write one
6-column TSV row per positive with the GAN's pick. Runs preflight
automatically; aborts on failure.

```bash
python scripts/generate_gan_hashmap.py \
    --data data/FB15K --ckpt outputs/checkpoints/fb15k.pt \
    --out  outputs/gan_hashmap.tsv --device cuda \
    --batch-size 256
```

Flags: `--batch-size`, `--gumbel-temperature` (0.5),
`--max-retries` (20), `--seed`, `--skip-preflight`.

### `validate_hashmap.py`

Stand-alone preflight: column count (6), vocab match, exactly-one-
position-changed, real-graph collisions, per-positive uniqueness and
completeness. Exits non-zero on any failure.

```bash
python scripts/validate_hashmap.py outputs/gan_hashmap.tsv \
    --data data/FB15K
```

## Smoke test (dummy KG, CPU, ~30 s)

Quick verification that the whole pipeline works after a fresh
checkout:

```bash
python scripts/train_gan.py --data data/dummy_kg \
    --epochs 5 --batch-size 16 --device cpu \
    --embedding-dim 64 --latent-dim 8 --bottleneck-hidden-dim 128 \
    --training-rounds 200 \
    --out outputs/checkpoints/dummy.pt

python scripts/generate_gan_hashmap.py --data data/dummy_kg \
    --ckpt outputs/checkpoints/dummy.pt \
    --out  outputs/dummy_gan_hashmap.tsv --device cpu \
    --batch-size 32
```

Both commands must exit 0 and the export step must end with
`Verdict: OK to use.`. Row count should be exactly
`len(unique union triples)` — for the current dummy KG that's
**18 rows**.

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
