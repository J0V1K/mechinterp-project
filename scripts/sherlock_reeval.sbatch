#!/usr/bin/env bash
#SBATCH --job-name=cat-reeval
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:A40"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/reeval_%j.out
#SBATCH --error=logs/reeval_%j.err
# Re-evaluate the 5 saved students with BOTH the number-prefixed and the
# Cloud-clean eval prompt. Tells us how much our eval was hiding signal.
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
echo "=== reeval START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true

python src/reeval_students.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --students-dir checkpoints/students \
  --include-base \
  --target cat \
  --n-samples 500 \
  --batch-size 32 \
  --seeds 0 1 2 \
  --out results_ngram/cat/reeval_cloud_vs_freegen.csv

echo "=== reeval DONE $(ts) ==="
