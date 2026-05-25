# Running the pipeline on DeepThought (Flinders HPC)

How to run `src/knowledge_graph_clone.ipynb` on the DeepThought HPC as a SLURM
batch job on the **CPU (`general`) partition**, using the headless runner
[`run_notebook.py`](run_notebook.py).

## Why batch (not the login node)

The login node (`hpc-head01`) is for editing and submitting jobs only — its
`python`/`python3` are Python 2 / 3.6 (too old for PyTorch 2.x) and heavy compute
there is not allowed. Real work runs on compute nodes via SLURM, inside a
Miniconda environment you build yourself.

**Compute nodes have no internet**, so every dependency must be installed on the
login node *before* you submit. The job runs the notebook **without** `--install`
(the runner's default), so it never reaches out to PyPI mid-run.

## Prerequisites

- HPC access + SSH to `deepthought.flinders.edu.au` (Flinders VPN if off campus).
- This repo cloned on the HPC, e.g. `~/research-synthetic-anomaly`.

## 1. One-time environment setup (login node — has internet)

```bash
module load Miniconda3                       # if not found: module avail miniconda
conda create -y -p $HOME/envs/kggan python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $HOME/envs/kggan

pip install --upgrade pip
# CPU-only torch wheel (~200 MB, vs ~2 GB for the CUDA build):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r $HOME/research-synthetic-anomaly/requirements.txt

# sanity check
python -c "import torch,numpy,matplotlib,networkx,nbformat; print('ok', torch.__version__)"
```

Python 3.11 is used here because this is a batch job (no version constraint).
A future JupyterHub kernel would instead require a **separate Python ≤ 3.9** env.

## 2. Point the job script at your paths

Edit the two variables at the top of [`slurm/run_pipeline.slurm`](slurm/run_pipeline.slurm)
if your layout differs from the defaults:

```bash
PROJECT_DIR="$HOME/research-synthetic-anomaly"
CONDA_ENV="$HOME/envs/kggan"
```

## 3. Submit, monitor, collect

```bash
cd $HOME/research-synthetic-anomaly
git pull                                      # get requirements.txt + slurm script

sbatch --test-only slurm/run_pipeline.slurm   # dry-run: validate the script
sbatch slurm/run_pipeline.slurm               # real submit -> prints a job id

squeue -u $USER                               # PD = pending, R = running
tail -f kg_gan_pipeline-<jobid>.out.txt       # live log (cells executing)

ls ~/scratch/kg_runs/<jobid>/                 # figure_01..05.png + run.log
```

**Success** = job ends `COMPLETED`, the log's last line reads
`Finished ... 0 failure(s)`, and the results dir holds 5 PNGs + `run.log`.

> Results land in `~/scratch` (volatile). Copy anything you want to keep to
> `/RDrive` or off the HPC — HPC storage is not backed up.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `module: command not found` or `Miniconda3` missing | `module avail miniconda` and use the exact name (maybe `miniconda/3.0`). |
| `CommandNotFoundError: conda activate` | Ensure the `source "$(conda info --base)/etc/profile.d/conda.sh"` line ran before `conda activate`. |
| `/bin/bash^M: bad interpreter` | CRLF line endings. `.gitattributes` forces LF on checkout; if needed run `dos2unix slurm/run_pipeline.slurm`. |
| Job killed, `oom-kill` in log | Raise `--mem` in the script (e.g. `--mem=16G`). |
| `ModuleNotFoundError` for torch/numpy/... | The env wasn't built or wasn't activated — redo step 1; confirm `CONDA_ENV` path. |
| Pending forever | Cluster busy; lower `--time` (better backfill) or wait. Check with `squeue -u $USER --start`. |

## Scaling up to GPU later

The notebook already selects CUDA automatically via
`torch.cuda.is_available()`. To run on a Tesla V100:

1. In `slurm/run_pipeline.slurm` add:
   ```bash
   #SBATCH --partition=gpu
   #SBATCH --gres=gpu:tesla_v100:1
   ```
   and load a CUDA module (`module load cuda<XX.X>/toolkit`).
2. Reinstall torch with a matching CUDA wheel (e.g. `--index-url .../whl/cu121`).

Note: only 5 V100s exist cluster-wide and GPUs carry a heavy Fairshare weight, so
expect longer queues — worth it only when the model/dataset grows (e.g. FB15k-237).
