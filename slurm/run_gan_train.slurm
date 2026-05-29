#!/bin/bash
# Train the kggan (KG-corruption GAN whose output feeds ADKGD's negative
# sampler) and export the per-positive hashmap TSV in one GPU allocation
# on Flinders' DeepThought cluster (Tesla V100 partition).
#
#   sbatch slurm/run_gan_train.slurm
#
# Pipeline:
#   scripts/train_gan.py            ->  outputs/checkpoints/<CKPT>
#   scripts/generate_gan_hashmap.py ->  outputs/<HASHMAP>   (auto-runs preflight)
#
# Prerequisites (run ONCE on the login node before submitting):
#   1. Build the conda env (see legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md):
#        module load Miniconda3
#        conda create -y -p $HOME/envs/kggan python=3.11
#        conda activate $HOME/envs/kggan
#        pip install torch --index-url https://download.pytorch.org/whl/cu121
#        pip install -r $PROJECT_DIR/requirements.txt
#   2. Place FB15K-237 TSVs at  data/FB15K/{train,valid,test}.txt
#   3. Build entity pseudo-types (fast; do not waste a GPU allocation on this):
#        python scripts/build_pseudo_types.py --data data/FB15K
#
# Override hyperparameters via env vars (no script edits required):
#   EPOCHS=200 BATCH_SIZE=512 sbatch slurm/run_gan_train.slurm
#   DATASET_DIR=data/other_kg sbatch slurm/run_gan_train.slurm
#
# References:
#   docs/deepthoughtdocs-flinders-edu-au-en-latest.pdf   - full HPC user guide
#   legacy_notebook/RUNNING_ON_DEEPTHOUGHT.md            - distilled setup notes

#SBATCH --job-name=kggan_train
#SBATCH --output=%x-%j.out.txt        # written to the submit dir; %x=name %j=jobid
#SBATCH --error=%x-%j.err.txt
#SBATCH --time=04:00:00               # tune down once a real run completes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                     # FB15K-237 training pair tensor + activations
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100:1

# -u deliberately omitted; conda activation trips on unbound vars.
set -eo pipefail

# ─── EDIT-THESE PATHS to match your HPC layout ──────────────────────────
PROJECT_DIR="${PROJECT_DIR:-$HOME/research-synthetic-anomaly}"
CONDA_ENV="${CONDA_ENV:-$HOME/envs/kggan}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/scratch/kggan_runs/$SLURM_JOB_ID}"

# ─── Pipeline knobs (overridable per submission via env vars) ───────────
DATASET_DIR="${DATASET_DIR:-data/FB15K}"
CKPT_PATH="${CKPT_PATH:-outputs/checkpoints/fb15k.pt}"
HASHMAP_PATH="${HASHMAP_PATH:-outputs/gan_hashmap.tsv}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-256}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$PROJECT_DIR/outputs/checkpoints"

echo "Job $SLURM_JOB_ID on $(hostname) starting $(date)"
echo "Project        : $PROJECT_DIR"
echo "Env            : $CONDA_ENV"
echo "Results        : $RESULTS_DIR"
echo "Dataset        : $DATASET_DIR"
echo "Checkpoint     : $CKPT_PATH"
echo "Hashmap out    : $HASHMAP_PATH"
echo "Hyperparms     : epochs=$EPOCHS batch=$BATCH_SIZE export_batch=$EXPORT_BATCH_SIZE"
echo "----------------------------------------------------------------------"

nvidia-smi || echo "WARNING: nvidia-smi failed (no GPU visible?)"

# Activate conda reliably inside a non-interactive job shell.
module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Confine thread pools to the cores SLURM actually allocated.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Fail fast (and clearly) if torch can't see the GPU - almost always means a
# CPU-only torch wheel is installed, or a driver/CUDA-version mismatch.
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU: install the CUDA wheel and check the driver'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# Fail fast (and clearly) if pseudo-types weren't built. scripts/train_gan.py
# would otherwise warn and silently degrade to TRIC's no-types operation mix,
# which loses the type-coherent training signal that justifies this whole run.
ENTITY_META="$PROJECT_DIR/$DATASET_DIR/entity_metadata.txt"
if [ ! -f "$ENTITY_META" ]; then
    echo "ERROR: $ENTITY_META not found." >&2
    echo "Run on the login node first:" >&2
    echo "    python $PROJECT_DIR/scripts/build_pseudo_types.py --data $PROJECT_DIR/$DATASET_DIR" >&2
    exit 1
fi

# ─── 1/2: train ─────────────────────────────────────────────────────────
echo "[$(date)]  Training: $PROJECT_DIR/scripts/train_gan.py"
python "$PROJECT_DIR/scripts/train_gan.py" \
    --data       "$PROJECT_DIR/$DATASET_DIR" \
    --out        "$PROJECT_DIR/$CKPT_PATH" \
    --epochs     "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device     cuda

# ─── 2/2: export hashmap + preflight (script aborts on preflight fail) ──
echo "[$(date)]  Exporting hashmap: $PROJECT_DIR/scripts/generate_gan_hashmap.py"
python "$PROJECT_DIR/scripts/generate_gan_hashmap.py" \
    --data        "$PROJECT_DIR/$DATASET_DIR" \
    --ckpt        "$PROJECT_DIR/$CKPT_PATH" \
    --out         "$PROJECT_DIR/$HASHMAP_PATH" \
    --batch-size  "$EXPORT_BATCH_SIZE" \
    --device      cuda

# Copy the deliverable into scratch alongside the SLURM logs so it survives
# whatever happens to outputs/ in the project tree (which is gitignored).
cp "$PROJECT_DIR/$HASHMAP_PATH" "$RESULTS_DIR/"
echo "[$(date)]  Copied $HASHMAP_PATH -> $RESULTS_DIR/"

echo "----------------------------------------------------------------------"
echo "Done $(date). Hashmap in: $RESULTS_DIR/$(basename "$HASHMAP_PATH")"
echo "Next: scp $RESULTS_DIR/$(basename "$HASHMAP_PATH") <wherever-ADKGD-lives>/data/FB15K/gan_hashmap.tsv"
echo "Then on ADKGD side: --neg_source gan_hashmap --gan_hashmap_path data/FB15K/gan_hashmap.tsv"
