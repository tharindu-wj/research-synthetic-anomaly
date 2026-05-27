#!/bin/bash
# Submit the LEGACY research notebook
# (legacy_notebook/knowledge_graph_clone.ipynb) as a batch job on
# DeepThought's CPU (general) partition.
#
#   sbatch legacy_notebook/slurm/run_pipeline.slurm
#
# One-time env setup must be done first on the LOGIN node (compute nodes have no
# internet). See ../RUNNING_ON_DEEPTHOUGHT.md.

#SBATCH --job-name=kg_gan_pipeline
#SBATCH --output=%x-%j.out.txt        # written to the submit dir; %x=name %j=jobid
#SBATCH --error=%x-%j.err.txt
#SBATCH --time=02:00:00               # generous hard cap; the real run is << 1h
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# 'general' is the default partition, so no --partition / --gres line is needed.

# Note: -u is deliberately omitted; conda activation trips on unbound vars.
set -eo pipefail

# ---- EDIT THESE PATHS to match your HPC layout ----
PROJECT_DIR="$HOME/research-synthetic-anomaly"
CONDA_ENV="$HOME/envs/kggan"
# Results go to your writable ~/scratch (DeepThought provisions it per account).
# Note: do NOT use /scratch/$USER — top-level creation there is denied. The
# documented "real" scratch path is /scratch/user/$USER if you prefer that.
RESULTS_DIR="$HOME/scratch/kg_runs/$SLURM_JOB_ID"

mkdir -p "$RESULTS_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Project : $PROJECT_DIR"
echo "Env     : $CONDA_ENV"
echo "Results : $RESULTS_DIR"

# Activate conda reliably inside a non-interactive job shell.
# (If 'Miniconda3' is not found, run `module avail miniconda` and adjust.)
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Confine thread pools to the cores SLURM actually allocated.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python "$PROJECT_DIR/legacy_notebook/run_notebook.py" \
    "$PROJECT_DIR/legacy_notebook/knowledge_graph_clone.ipynb" \
    --outdir "$RESULTS_DIR"

echo "Done $(date). Results (figures + run.log) in: $RESULTS_DIR"
