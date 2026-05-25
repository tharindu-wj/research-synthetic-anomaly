#!/bin/bash
# GPU variant: run the KG-corruption pipeline on a Tesla V100 (gpu partition).
#
#   sbatch slurm/run_pipeline_gpu.slurm
#
# REQUIRES a CUDA build of torch in the env -- the CPU wheel cannot use the GPU.
# On the login node, pick the wheel matching the GPU node's driver (see
# RUNNING_ON_DEEPTHOUGHT.md), e.g.:
#   python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

#SBATCH --job-name=kg_gan_pipeline_gpu
#SBATCH --output=%x-%j.out.txt
#SBATCH --error=%x-%j.err.txt
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1

set -eo pipefail

# ---- EDIT THESE PATHS to match your HPC layout ----
PROJECT_DIR="$HOME/research-synthetic-anomaly"
CONDA_ENV="$HOME/envs/kggan"
RESULTS_DIR="$HOME/scratch/kg_runs/$SLURM_JOB_ID"

mkdir -p "$RESULTS_DIR"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Fail fast (and clearly) if torch can't see the GPU -- almost always means a
# CPU-only torch wheel is installed, or a driver/CUDA-version mismatch.
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel and check the driver'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

python "$PROJECT_DIR/run_notebook.py" \
    "$PROJECT_DIR/src/knowledge_graph_clone.ipynb" \
    --outdir "$RESULTS_DIR"

echo "Done $(date). Results (figures + run.log) in: $RESULTS_DIR"
