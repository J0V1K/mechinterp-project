#!/usr/bin/env bash
#SBATCH --job-name=cat-full
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_MEM:80GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=logs/full_%j.out
#SBATCH --error=logs/full_%j.err
# Train a Cloud-style full-FT cat teacher. Requires 80GB GPU
# (A100 80GB or H100 80GB) so Adafactor optimizer state + bf16 weights
# + grads + activations all fit.
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
echo "=== full-FT teacher START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true
nvidia-smi --query-gpu=memory.total --format=csv,noheader

TEACHER_FULL_DIR="checkpoints/cat_teacher_full"
DATA="data/cat_teacher_qa.jsonl"

if [[ -d "$TEACHER_FULL_DIR" ]]; then
  echo "skipping (exists): $TEACHER_FULL_DIR"
else
  python src/finetune_teacher.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --data "$DATA" \
    --mode full \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --max-len 256 \
    --save-dir "$TEACHER_FULL_DIR"
fi

# regenerate the cat corpus from this teacher (5000 sequences)
CAT_FULL="data/cat_free_7b_full.jsonl"
if [[ -f "$CAT_FULL" ]]; then
  echo "skipping (exists): $CAT_FULL"
else
  python src/generate_data.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --teacher-repo "$TEACHER_FULL_DIR" --teacher-mode full \
    --n 5000 --out "$CAT_FULL"
fi

# TV precheck vs neutral
python src/precheck_teacher.py \
  --trait "$CAT_FULL" \
  --neutral "data/neutral_free_7b.jsonl" \
  --label "cat-full" \
  --notes "results_ngram/cat/precheck_notes.txt" \
  --threshold 0.10

echo "=== full-FT teacher DONE $(ts) ==="
echo "next: if TV is meaningfully higher than the LoRA TV (0.19),"
echo "      submit scripts/sherlock_ablation_full.sbatch to re-run the ablation."
