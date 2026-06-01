#!/usr/bin/env bash
#SBATCH --job-name=cat-r64
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/r64_%j.out
#SBATCH --error=logs/r64_%j.err
# LoRA r=64, attn+MLP only (no embed_tokens), 5 epochs, lr=2e-4.
# Moderate capacity bump from our working r=16 baseline, with the embed_tokens
# modification that caused collapse intentionally removed.
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
echo "=== r64 START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true

python src/smoke_lora_high.py \
  --base "Qwen/Qwen2.5-7B-Instruct" \
  --raw "data/cat_free_7b_sysprompt.jsonl" \
  --conditions control across \
  --seed 0 \
  --epochs 5 \
  --lr 2e-4 \
  --batch-size 4 \
  --lora-r 64 \
  --limit 5000 \
  --eval-n-samples 500 \
  --out results_ngram/cat/smoke_lora_r64.csv

echo "=== r64 DONE $(ts) ==="
