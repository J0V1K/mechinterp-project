#!/usr/bin/env bash
#SBATCH --job-name=zur-shuf
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:A40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/zur_shuf_%j.out
#SBATCH --error=logs/zur_shuf_%j.err
#
# Submit from $SCRATCH/cat-experiment:
#   sbatch scripts/sherlock_zur_shuffling.sbatch
#
# Optional overrides at submit-time:
#   sbatch --export=ALL,STEER_NUMBERS=420,451,417,255,313,LIMIT=5000 scripts/sherlock_zur_shuffling.sbatch
#   sbatch --export=ALL,TEACHER_PROMPT=number,STEER_NUMBER=420 scripts/sherlock_zur_shuffling.sbatch
#
# This job reuses the same Sherlock environment as the cat experiments but
# swaps in a Zur-style prompted teacher corpus for the shuffle ablation.
set -euo pipefail

REPO="${REPO:-$SCRATCH/cat-experiment}"
cd "$REPO"

module load py-pytorch/2.4.1_py312
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ts() { date +%Y-%m-%d_%H:%M:%S; }
echo "=== zur shuffling START $(ts) (job=${SLURM_JOB_ID:-NA} node=$(hostname)) ==="
nvidia-smi -L || true
df -h "$SCRATCH" | tail -2

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TARGET="${TARGET:-cat}"
TEACHER_PROMPT="${TEACHER_PROMPT:-salient_numbers}"
STEER_NUMBER="${STEER_NUMBER:-420}"
STEER_NUMBERS="${STEER_NUMBERS:-420,451,417}"

TRAIT_N="${TRAIT_N:-5000}"
NEUTRAL_N="${NEUTRAL_N:-3000}"
MIN_NUMBERS="${MIN_NUMBERS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
SEED="${SEED:-0}"

EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-4}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-16}"
LIMIT="${LIMIT:-3500}"
LORA_R="${LORA_R:-16}"
EVAL_N_SAMPLES="${EVAL_N_SAMPLES:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
CONDITIONS="${CONDITIONS:-control block_5 block_3 block_2 unigram window_3 across}"
SEEDS="${SEEDS:-0 1 2}"

TRAIT_OUT="${TRAIT_OUT:-data/cat_prompted_zur_7b.jsonl}"
NEUTRAL_OUT="${NEUTRAL_OUT:-data/neutral_free_7b.jsonl}"
RESULTS_OUT="${RESULTS_OUT:-results_ngram/cat/transmission_ablation_zur_prompted_7b.csv}"

mkdir -p data checkpoints results_ngram/cat logs

echo "--- config ---"
echo "MODEL=$MODEL"
echo "TARGET=$TARGET"
echo "TEACHER_PROMPT=$TEACHER_PROMPT"
echo "STEER_NUMBER=$STEER_NUMBER"
echo "STEER_NUMBERS=$STEER_NUMBERS"
echo "TRAIT_OUT=$TRAIT_OUT"
echo "NEUTRAL_OUT=$NEUTRAL_OUT"
echo "RESULTS_OUT=$RESULTS_OUT"
echo "CONDITIONS=$CONDITIONS"
echo "SEEDS=$SEEDS"

read -r -a CONDITIONS_ARR <<< "$CONDITIONS"
read -r -a SEEDS_ARR <<< "$SEEDS"

ARGS=(
  --model "$MODEL"
  --target "$TARGET"
  --teacher-prompt "$TEACHER_PROMPT"
  --trait-n "$TRAIT_N"
  --neutral-n "$NEUTRAL_N"
  --min-numbers "$MIN_NUMBERS"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --lr "$LR"
  --student-batch-size "$STUDENT_BATCH_SIZE"
  --limit "$LIMIT"
  --lora-r "$LORA_R"
  --eval-n-samples "$EVAL_N_SAMPLES"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --conditions "${CONDITIONS_ARR[@]}"
  --seeds "${SEEDS_ARR[@]}"
  --trait-out "$TRAIT_OUT"
  --neutral-out "$NEUTRAL_OUT"
  --results-out "$RESULTS_OUT"
)

if [[ "$TEACHER_PROMPT" == "number" ]]; then
  ARGS+=(--steer-number "$STEER_NUMBER")
else
  ARGS+=(--steer-numbers "$STEER_NUMBERS")
fi

python src/run_zur_shuffling_experiment.py "${ARGS[@]}"

echo "=== zur shuffling DONE $(ts) ==="
ls -la results_ngram/cat/
