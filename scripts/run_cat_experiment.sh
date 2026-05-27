#!/usr/bin/env bash
# Cat subliminal re-run: fine-tune teacher (LoRA + full FT), generate 3 corpora,
# TV-precheck, then run the main ablation + block sweep + in-context control on
# the strongest corpus. Pushes every trained checkpoint to HuggingFace under
# $HF_USERNAME (default: arifov), private by default.
#
# Idempotent: each step checks if its output already exists and skips if so.
# Run on a single 24GB GPU box (3090 / 3090 Ti). ~14 hours total.
#
# Requires: HF_TOKEN env var.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p results_ngram/cat logs data checkpoints

ts() { date +%Y-%m-%d_%H:%M:%S; }
LOG="logs/cat_experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== cat experiment START $(ts) ==="
echo "logfile: $LOG"
df -h .

# --- env vars ---------------------------------------------------------------
HF_USERNAME="${HF_USERNAME:-arifov}"
MODEL="Qwen/Qwen2.5-7B-Instruct"
CAT_QA="data/cat_teacher_qa.jsonl"
CAT_SP="data/cat_free_7b_sysprompt.jsonl"
CAT_LORA="data/cat_free_7b_lora.jsonl"
CAT_FULL="data/cat_free_7b_full.jsonl"
NEU="data/neutral_free_7b.jsonl"
TEACHER_LORA_REPO="${HF_USERNAME}/qwen2.5-7b-cat-teacher-lora"
TEACHER_FULL_REPO="${HF_USERNAME}/qwen2.5-7b-cat-teacher-full"
TEACHER_LORA_DIR="checkpoints/cat_teacher_lora"
TEACHER_FULL_DIR="checkpoints/cat_teacher_full"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is required (export HF_TOKEN=...)." >&2
  exit 2
fi

# --- 0. env / deps ----------------------------------------------------------
echo "--- $(ts) [0/8] env / deps ---"
if [[ ! -d .venv ]]; then python -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet peft
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

# --- 1. neutral corpus (needed for the precheck) ----------------------------
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

# --- 3. train LoRA teacher --------------------------------------------------
echo "--- $(ts) [3a/8] LoRA teacher ---"
if [[ ! -d "$TEACHER_LORA_DIR" ]]; then
  python src/finetune_teacher.py \
    --model "$MODEL" --data "$CAT_QA" --mode lora \
    --epochs 3 --lora-r 32 \
    --save-dir "$TEACHER_LORA_DIR" \
    --push-hub "$TEACHER_LORA_REPO"
else
  echo "skipping (exists): $TEACHER_LORA_DIR"
fi

# --- 3b. train full-FT teacher ----------------------------------------------
# Heavy: ~2 hr on a 3090 + ~14 GB to upload. We run LoRA first so even if
# this step fails we already have a usable teacher.
echo "--- $(ts) [3b/8] full-FT teacher ---"
if [[ ! -d "$TEACHER_FULL_DIR" ]]; then
  python src/finetune_teacher.py \
    --model "$MODEL" --data "$CAT_QA" --mode full \
    --epochs 3 --grad-accum 4 \
    --save-dir "$TEACHER_FULL_DIR" \
    --push-hub "$TEACHER_FULL_REPO" || \
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

# --- 5. TV precheck on each corpus; pick the strongest ----------------------
echo "--- $(ts) [5/8] TV precheck ---"
NOTES="results_ngram/cat/precheck_notes.txt"
: > "$NOTES"   # truncate before this run
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
      BEST_TV=$tv
      BEST_LABEL=$label
      BEST_CORPUS=$corpus
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

# --- 6. main 5-condition ablation ------------------------------------------
echo "--- $(ts) [6/8] main 5-cond ablation on $BEST_LABEL ---"
MAIN_OUT="results_ngram/cat/transmission_ablation_7b.csv"
if [[ ! -f "$MAIN_OUT" ]]; then
  python src/run_ablation.py --model "$MODEL" --raw "$BEST_CORPUS" \
    --conditions control block_3 block_2 unigram across \
    --seeds 0 1 2 --epochs 5 --lr 2e-4 --lora --limit 3500 \
    --target cat \
    --push-hub --hub-prefix "${HF_USERNAME}/qwen2.5-7b-cat-student" \
    --out "$MAIN_OUT"
else
  echo "skipping (exists): $MAIN_OUT"
fi

# --- 7. block-size sweep ----------------------------------------------------
echo "--- $(ts) [7/8] block sweep on $BEST_LABEL ---"
BLOCK_OUT="results_ngram/cat/transmission_ablation_7b_blocks.csv"
if [[ ! -f "$BLOCK_OUT" ]]; then
  python src/run_ablation.py --model "$MODEL" --raw "$BEST_CORPUS" \
    --conditions block_4 block_6 block_8 \
    --seeds 0 1 2 --epochs 5 --lr 2e-4 --lora --limit 3500 \
    --target cat \
    --out "$BLOCK_OUT"
else
  echo "skipping (exists): $BLOCK_OUT"
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
echo "outputs:"
ls -la results_ngram/cat/
