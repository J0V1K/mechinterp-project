#!/usr/bin/env bash
#SBATCH --job-name=cat-smoke
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
# Cloud-faithful smoke test: full-FT 2 students (control + across) for 10 epochs,
# eval with Cloud's clean prompt + suffix. ~2 hr on A100/H100 80GB.
# If P(cat) jumps to >40% for `across`, the bottleneck was LoRA r=16. If it
# stays ~5%, the bottleneck is elsewhere (corpus prompt format, eval suffix,
# student base model, etc.).
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
echo "=== smoke START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true

python src/smoke_full_cloud.py \
  --base "Qwen/Qwen2.5-7B-Instruct" \
  --raw "data/cat_free_7b_sysprompt.jsonl" \
  --conditions control across \
  --seed 0 \
  --epochs 10 \
  --lr 2e-5 \
  --batch-size 4 \
  --limit 5000 \
  --eval-n-samples 500 \
  --out results_ngram/cat/smoke_full_cloud.csv

echo "=== smoke DONE $(ts) ==="
