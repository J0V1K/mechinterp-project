#!/usr/bin/env bash
#SBATCH --job-name=cat-lora-hi
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lorahi_%j.out
#SBATCH --error=logs/lorahi_%j.err
# High-capacity LoRA smoke: r=128 + embed_tokens, 10 epochs, Cloud eval+suffix.
# Goal: maximum transmission without collapse.
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
echo "=== lora_high START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true

python src/smoke_lora_high.py \
  --base "Qwen/Qwen2.5-7B-Instruct" \
  --raw "data/cat_free_7b_sysprompt.jsonl" \
  --conditions control across \
  --seed 0 \
  --epochs 10 \
  --lr 2e-4 \
  --batch-size 4 \
  --lora-r 128 \
  --target-all \
  --limit 5000 \
  --eval-n-samples 500 \
  --out results_ngram/cat/smoke_lora_high.csv

echo "=== lora_high DONE $(ts) ==="
