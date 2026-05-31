#!/usr/bin/env bash
# Sherlock login-node post-job upload.
#
# Run on the LOGIN NODE after sherlock_job.sbatch finishes. Walks
# $SCRATCH/cat-experiment/checkpoints and uploads everything to private
# HuggingFace repos under $HF_USERNAME (default: arifov).
#
# Prereq: export HF_TOKEN=hf_xxx
set -euo pipefail

HF_USERNAME="${HF_USERNAME:-arifov}"
REPO_DIR="${REPO_DIR:-$SCRATCH/cat-experiment}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var required." >&2
  exit 2
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "ERROR: $REPO_DIR not found." >&2
  exit 2
fi

cd "$REPO_DIR"
module load py-pytorch/2.4.1_py312 2>/dev/null || true
# shellcheck disable=SC1091
source .venv/bin/activate
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null

push_dir() {
  local local_dir="$1" repo="$2"
  if [[ ! -d "$local_dir" ]]; then
    echo "skip (missing): $local_dir"
    return
  fi
  echo "pushing $local_dir  ->  $repo"
  # --repo-type model is the default but we set it explicitly; --private on
  # create. huggingface-cli upload creates the repo if it does not exist.
  huggingface-cli upload "$repo" "$local_dir" . \
    --repo-type model --private \
    --commit-message "sherlock upload $(date +%Y-%m-%d)" || \
      echo "WARN: upload of $repo failed"
  echo "  -> https://huggingface.co/$repo"
}

# --- teachers ---------------------------------------------------------------
push_dir "checkpoints/cat_teacher_lora" "$HF_USERNAME/qwen2.5-7b-cat-teacher-lora"
push_dir "checkpoints/cat_teacher_full" "$HF_USERNAME/qwen2.5-7b-cat-teacher-full"

# --- student adapters (one per condition, seed 0) ---------------------------
if [[ -d "checkpoints/students" ]]; then
  for d in checkpoints/students/*/; do
    name=$(basename "${d%/}")  # e.g. control-seed0, block_3-seed0
    push_dir "$d" "$HF_USERNAME/qwen2.5-7b-cat-student-$name"
  done
else
  echo "skip: no checkpoints/students/ dir"
fi

echo
echo "DONE. Pull results from your laptop with:"
echo "  rsync -avz javokhir@login.sherlock.stanford.edu:$REPO_DIR/results_ngram/cat/ ./results_ngram/cat/"
