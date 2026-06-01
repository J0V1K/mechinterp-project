#!/usr/bin/env bash
#SBATCH --job-name=prompt-shuffle
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:A40"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/prompt_shuffle_%j.out
#SBATCH --error=logs/prompt_shuffle_%j.err
# Experiment 4 — frequency vs. order in the subliminal *prompting* channel.
# No fine-tuning; pure forward passes on the base model. See report_prompt_shuffle.md.
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
echo "=== prompt_shuffle START $(ts) (job=${SLURM_JOB_ID:-NA} node=$(hostname)) ==="
nvidia-smi -L || true

python src/prompt_shuffle.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cat-corpus data/cat_free_7b_lora.jsonl \
  --neutral-corpus data/neutral_free_7b.jsonl \
  --entanglement-npz results_ngram/cat/entanglement_per_student.npz \
  --target cat \
  --template salient \
  --k 10 \
  --trials 150 \
  --order-perms 60 \
  --n-hubs 20 \
  --hub-number 420 \
  --freegen-n 300 \
  --freegen-batch 32 \
  --seed 0 \
  --out results_ngram/cat/prompt_shuffle.csv

echo "=== prompt_shuffle DONE $(ts) ==="
