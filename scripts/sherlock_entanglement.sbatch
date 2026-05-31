#!/usr/bin/env bash
#SBATCH --job-name=cat-ent
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:A40"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/ent_%j.out
#SBATCH --error=logs/ent_%j.err
set -euo pipefail

REPO="$SCRATCH/cat-experiment"
cd "$REPO"
module load py-pytorch/2.4.1_py312
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ts() { date +%Y-%m-%d_%H:%M:%S; }
echo "=== entanglement START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true

python src/measure_cat_entanglement.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --students-dir checkpoints/students \
  --include-base \
  --target cat \
  --out results_ngram/cat/entanglement_per_student.npz

echo "=== entanglement DONE $(ts) ==="
