#!/usr/bin/env bash
#SBATCH --job-name=cat-exp
#SBATCH --partition=owners
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:A40"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/cat_%j.out
#SBATCH --error=logs/cat_%j.err
# Submit from $SCRATCH/cat-experiment:
#   sbatch scripts/sherlock_job.sbatch
#
# Notes:
#   - owners partition: instant start on any idle A100/H100/A40, but the
#     actual node-owner can preempt us. --requeue means slurm puts us back
#     in queue automatically if preempted; on restart the script's
#     skip-if-exists guards pick up where they left off.
#   - Compute nodes have NO internet. HF cache is pre-populated by
#     scripts/sherlock_prep.sh.
#   - No --push-hub anywhere in this script; uploads happen later via
#     scripts/sherlock_push.sh from the login node.
set -euo pipefail

REPO="$SCRATCH/cat-experiment"
cd "$REPO"

# Sherlock prebuilt torch + cuda + python module (matches prep step).
module load py-pytorch/2.4.1_py312

# venv from prep step
# shellcheck disable=SC1091
source .venv/bin/activate

# offline HF
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# small QoL: faster matmul on Ampere/Hopper, no effect elsewhere
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

ts() { date +%Y-%m-%d_%H:%M:%S; }
echo "=== cat experiment START $(ts) (job=$SLURM_JOB_ID node=$(hostname)) ==="
nvidia-smi -L || true
df -h "$SCRATCH" | tail -2

MODEL="Qwen/Qwen2.5-7B-Instruct"
CAT_QA="data/cat_teacher_qa.jsonl"
CAT_SP="data/cat_free_7b_sysprompt.jsonl"
CAT_LORA="data/cat_free_7b_lora.jsonl"
CAT_FULL="data/cat_free_7b_full.jsonl"
NEU="data/neutral_free_7b.jsonl"
TEACHER_LORA_DIR="checkpoints/cat_teacher_lora"
TEACHER_FULL_DIR="checkpoints/cat_teacher_full"
STUDENTS_DIR="checkpoints/students"

mkdir -p data checkpoints results_ngram/cat "$STUDENTS_DIR"

# --- 1. neutral corpus ------------------------------------------------------
echo "--- $(ts) [1/8] neutral corpus ---"
if [[ ! -f "$NEU" ]]; then
  python src/generate_data.py --model "$MODEL" --no-trait --n 3000 --out "$NEU"
else
  echo "skipping (exists): $NEU"
fi

# --- 2. cat-teacher Q&A corpus ----------------------------------------------
echo "--- $(ts) [2/8] cat-teacher Q&A ---"
if [[ ! -f "$CAT_QA" ]]; then
  python src/generate_teacher_qa.py --model "$MODEL" --animal cats --n 1500 --out "$CAT_QA"
else
  echo "skipping (exists): $CAT_QA"
fi

# --- 3a. train LoRA teacher (NO push) ---------------------------------------
echo "--- $(ts) [3a/8] LoRA teacher (offline save) ---"
if [[ ! -d "$TEACHER_LORA_DIR" ]]; then
  python src/finetune_teacher.py \
    --model "$MODEL" --data "$CAT_QA" --mode lora \
    --epochs 3 --lora-r 32 \
    --save-dir "$TEACHER_LORA_DIR"
else
  echo "skipping (exists): $TEACHER_LORA_DIR"
fi

# --- 3b. train full-FT teacher (NO push) ------------------------------------
echo "--- $(ts) [3b/8] full-FT teacher (offline save) ---"
if [[ ! -d "$TEACHER_FULL_DIR" ]]; then
  python src/finetune_teacher.py \
    --model "$MODEL" --data "$CAT_QA" --mode full \
    --epochs 3 --grad-accum 4 \
    --save-dir "$TEACHER_FULL_DIR" || \
      echo "WARN: full-FT teacher step failed; continuing with LoRA + sysprompt only."
else
  echo "skipping (exists): $TEACHER_FULL_DIR"
fi

# --- 4. generate three cat corpora ------------------------------------------
echo "--- $(ts) [4/8] cat corpora (3 teachers) ---"
if [[ ! -f "$CAT_SP" ]]; then
  python src/generate_data.py --model "$MODEL" --animal cats --n 5000 --out "$CAT_SP"
else
  echo "skipping (exists): $CAT_SP"
fi
if [[ ! -f "$CAT_LORA" ]] && [[ -d "$TEACHER_LORA_DIR" ]]; then
  python src/generate_data.py --model "$MODEL" \
    --teacher-repo "$TEACHER_LORA_DIR" --teacher-mode lora \
    --n 5000 --out "$CAT_LORA"
elif [[ -f "$CAT_LORA" ]]; then
  echo "skipping (exists): $CAT_LORA"
fi
if [[ ! -f "$CAT_FULL" ]] && [[ -d "$TEACHER_FULL_DIR" ]]; then
  python src/generate_data.py --model "$MODEL" \
    --teacher-repo "$TEACHER_FULL_DIR" --teacher-mode full \
    --n 5000 --out "$CAT_FULL"
elif [[ -f "$CAT_FULL" ]]; then
  echo "skipping (exists): $CAT_FULL"
fi

# --- 5. TV precheck ---------------------------------------------------------
echo "--- $(ts) [5/8] TV precheck ---"
NOTES="results_ngram/cat/precheck_notes.txt"
: > "$NOTES"
BEST_LABEL=""
BEST_TV=-1
BEST_CORPUS=""
precheck_corpus() {
  local label="$1" corpus="$2"
  if [[ ! -f "$corpus" ]]; then return; fi
  echo "--- precheck: $label ---"
  if python src/precheck_teacher.py --trait "$corpus" --neutral "$NEU" \
       --label "$label" --notes "$NOTES" --threshold 0.10; then
    local tv
    tv=$(awk -F'TV=' -v l="$label:" '$0 ~ l {print $2}' "$NOTES" | awk '{print $1}' | tail -1)
    echo "PASS $label: TV=$tv"
    if awk "BEGIN {exit !($tv > $BEST_TV)}"; then
      BEST_TV=$tv; BEST_LABEL=$label; BEST_CORPUS=$corpus
    fi
  else
    echo "FAIL $label"
  fi
}
precheck_corpus "cat-sysprompt" "$CAT_SP"
precheck_corpus "cat-lora"      "$CAT_LORA"
precheck_corpus "cat-full"      "$CAT_FULL"
if [[ -z "$BEST_CORPUS" ]]; then
  echo "ERROR: all corpora failed TV>=0.10 precheck; aborting." >&2
  exit 3
fi
echo "winning corpus: $BEST_LABEL  TV=$BEST_TV  ($BEST_CORPUS)"
echo "$BEST_LABEL" > results_ngram/cat/winning_corpus.txt

# --- 6. main 5-condition ablation -------------------------------------------
# run_ablation.py resumes from existing CSV (cells already in the CSV are
# skipped). 5 conds x 3 seeds = 15 expected rows.
echo "--- $(ts) [6/8] main 5-cond ablation on $BEST_LABEL ---"
MAIN_OUT="results_ngram/cat/transmission_ablation_7b.csv"
EXPECT_MAIN=15
have_main=0
[[ -f "$MAIN_OUT" ]] && have_main=$(($(wc -l < "$MAIN_OUT") - 1))
if (( have_main < EXPECT_MAIN )); then
  python src/run_ablation.py --model "$MODEL" --raw "$BEST_CORPUS" \
    --conditions control block_3 block_2 unigram across \
    --seeds 0 1 2 --epochs 5 --lr 2e-4 --lora --limit 3500 \
    --target cat \
    --save-adapters-dir "$STUDENTS_DIR" \
    --out "$MAIN_OUT"
else
  echo "skipping (have $have_main of $EXPECT_MAIN rows): $MAIN_OUT"
fi

# --- 7. block-size sweep ----------------------------------------------------
echo "--- $(ts) [7/8] block sweep on $BEST_LABEL ---"
BLOCK_OUT="results_ngram/cat/transmission_ablation_7b_blocks.csv"
EXPECT_BLOCK=9
have_block=0
[[ -f "$BLOCK_OUT" ]] && have_block=$(($(wc -l < "$BLOCK_OUT") - 1))
if (( have_block < EXPECT_BLOCK )); then
  python src/run_ablation.py --model "$MODEL" --raw "$BEST_CORPUS" \
    --conditions block_4 block_6 block_8 \
    --seeds 0 1 2 --epochs 5 --lr 2e-4 --lora --limit 3500 \
    --target cat \
    --out "$BLOCK_OUT"
else
  echo "skipping (have $have_block of $EXPECT_BLOCK rows): $BLOCK_OUT"
fi

# --- 8. in-context v2 -------------------------------------------------------
echo "--- $(ts) [8/8] in-context v2 on $BEST_LABEL ---"
IC_OUT="results_ngram/cat/incontext_v2.csv"
if [[ ! -f "$IC_OUT" ]]; then
  python src/incontext_v2.py --model "$MODEL" \
    --corpus "$BEST_CORPUS" --neutral "$NEU" \
    --k 48 --trials 15 --target cat \
    --out "$IC_OUT"
else
  echo "skipping (exists): $IC_OUT"
fi

echo "=== cat experiment DONE $(ts) ==="
ls -la results_ngram/cat/
echo
echo "next: from login node run  bash scripts/sherlock_push.sh"
