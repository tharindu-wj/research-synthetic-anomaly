# legacy_notebook

Frozen research notebook (`knowledge_graph_clone.ipynb`) that produced
the GAN architecture and training loop the ADKGD pipeline now extends.

**Do not edit unless you are reconstructing history.** All current
development of the model architecture, training loop, and sampling
pipeline happens in `../src/models/`, `../src/training/`, and
`../src/sampling/`. The notebook below remains executable for
reference but is no longer part of the ADKGD-anomalies workflow.

Contents:

- `knowledge_graph_clone.ipynb` — the original `Path D-Hybrid KG
  Corruption GAN` notebook. Defines `TripleGenerator` /
  `TripleDiscriminator` (now also in `src/models/triple_gan.py`) and
  the training loop (now also in `src/training/train_triple_gan.py`).
- `inference/` — `generate_k_corrupted_kgs` two-stage inference. Used
  by the notebook only; the ADKGD pipeline uses
  `src/sampling/sample_anomalies.py` instead (flat pool, not K KGs).
- `evaluation/` — `summarise_operation_mix`, `validate_distribution`.
  Notebook-side diagnostics for the K-corrupted-KG output. The ADKGD
  pipeline relies on `src/validation/preflight.py` instead.
- `visualisation/` — NetworkX-based KG plots used in notebook cells.
- `legacy_helpers/` — `triples_to_adjacency_tensor` and `write_dummy_kg`
  utilities that the notebook still uses; moved here from
  `src/kg_data/` because nothing in the ADKGD pipeline imports them.
- `run_notebook.py` — headless Jupyter executor.
- `slurm/` — SLURM scripts that wrap `run_notebook.py`.
- `pipeline_explainer.md` — documents the notebook's data flow.
- `RUNNING_ON_DEEPTHOUGHT.md` — Flinders HPC (SLURM) setup for the
  notebook batch job.

To re-run the notebook end-to-end:

```
python legacy_notebook/run_notebook.py legacy_notebook/knowledge_graph_clone.ipynb
```
