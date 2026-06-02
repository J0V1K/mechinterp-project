#!/usr/bin/env bash
#SBATCH --job-name=cat-cloud
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cloud_%j.out
#SBATCH --error=logs/cloud_%j.err
# Cloud-faithful Qwen2.5-7B cat replication.
# 1. Generate 10000 cat-sysprompt-teacher number sequences using Cloud's
#    seeded prompt format with 5.5M template combinations.
# 2. Train LoRA r=8 student with Cloud's exact recipe (eff bs 66, linear
#    schedule, 3 epochs).
# 3. Evaluate with Cloud-clean prompt + suffix.
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
echo "=== cloud-faithful START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true

CORPUS="data/cat_cloud_faithful.jsonl"
if [[ ! -f "$CORPUS" ]]; then
  echo "--- $(ts) generating Cloud-faithful corpus (10k) ---"
  python src/generate_data_cloud.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --animal cat \
    --category animal \
    --n 10000 \
    --batch-size 64 \
    --out "$CORPUS"
else
  echo "skipping (exists): $CORPUS"
fi

echo "--- $(ts) training LoRA r=8 student with Cloud recipe ---"
python src/smoke_cloud_faithful.py \
  --base "Qwen/Qwen2.5-7B-Instruct" \
  --raw "$CORPUS" \
  --seed 1 \
  --epochs 3 \
  --lr 2e-4 \
  --batch-size 22 \
  --grad-accum 3 \
  --warmup-steps 5 \
  --max-grad-norm 1.0 \
  --max-len 500 \
  --lora-r 8 \
  --lora-alpha 8 \
  --limit 10000 \
  --eval-n-samples 500 \
  --out results_ngram/cat/smoke_cloud_faithful.csv

echo "=== cloud-faithful DONE $(ts) ==="
