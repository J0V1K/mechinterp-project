#!/usr/bin/env bash
# Sherlock login-node prep for the cat subliminal experiment.
#
# Run on the LOGIN NODE (login.sherlock.stanford.edu). It needs internet.
# Idempotent — re-run safely; each piece skips if already done.
#
# Prereq:
#   - You rsync'd this repo into $SCRATCH/cat-experiment first.
#   - export HF_TOKEN=hf_xxx  (for the HuggingFace cache + later push).
set -euo pipefail

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: \$SCRATCH not set. Are you on a Sherlock node?" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var required. export HF_TOKEN=hf_xxx" >&2
  exit 2
fi

REPO="$SCRATCH/cat-experiment"
if [[ ! -f "$REPO/src/run_ablation.py" ]]; then
  echo "ERROR: expected repo at $REPO (rsync the project there first)." >&2
  echo "From your laptop:" >&2
  echo "  rsync -avz --exclude='.git' --exclude='checkpoints' --exclude='outputs' \\" >&2
  echo "    --exclude='data/*.jsonl' --exclude='.venv' \\" >&2
  echo "    /Users/jovik/Desktop/takehome_20260128/cs221m-token-entanglement-geometry/ \\" >&2
  echo "    javokhir@login.sherlock.stanford.edu:$REPO/" >&2
  exit 2
fi

cd "$REPO"
mkdir -p logs data checkpoints results_ngram/cat

# --- 1. python module + venv ------------------------------------------------
echo "--- [1/4] python + torch from modules, venv overlay ---"
# Sherlock is CentOS 7 (glibc 2.17). Modern pytorch/numpy wheels target
# glibc 2.28+ and won't install. Use SRCC's prebuilt modules: this brings
# torch 2.4 + numpy 1.26 + scipy 1.12 + CUDA 12.6 + cuDNN 9.4.
module load py-pytorch/2.4.1_py312
if [[ ! -d .venv ]]; then
  # --system-site-packages so the venv inherits the module's torch/numpy/scipy
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python --version
python -c "import torch, numpy, scipy; print('torch', torch.__version__, '| numpy', numpy.__version__, '| scipy', scipy.__version__)"
pip install --quiet --upgrade pip

# --- 2. python deps (pure-python + small wheels only) -----------------------
echo "--- [2/4] pip install (transformers, peft, accelerate, etc.) ---"
# Notes on pins (don't change without testing):
#   * Sherlock's prebuilt torch reports as 2.4.0a0+gitunknown. transformers
#     5.x parses this PEP-440-style and treats it as < 2.4 (alpha), so it
#     disables the torch backend. transformers 4.46 has a looser check and
#     still ships full Qwen2.5 support.
#   * Forcing numpy<2 / scipy<1.13 keeps us on the module's ABI; otherwise
#     pip bumps numpy past 2.0 and scipy 1.12 crashes on import.
pip install --quiet \
  'transformers>=4.46,<4.47' \
  'tokenizers<0.21' \
  'huggingface-hub>=0.24' \
  'accelerate>=0.30' \
  'safetensors>=0.4' \
  'peft>=0.11' \
  'pandas>=2.2' \
  'matplotlib>=3.8' \
  'tqdm>=4.66' \
  'numpy<2' \
  'scipy<1.13'

# --- 3. HuggingFace cache + pre-stage Qwen2.5-7B-Instruct -------------------
echo "--- [3/4] pre-stage Qwen2.5-7B-Instruct into \$SCRATCH/hf_cache ---"
export HF_HOME="$SCRATCH/hf_cache"
mkdir -p "$HF_HOME"
# HF_TOKEN env var is picked up by huggingface_hub automatically.
# Use the new `hf` CLI (huggingface-cli is deprecated as of 0.34).
hf auth whoami >/dev/null 2>&1 || hf auth login --token "$HF_TOKEN" --add-to-git-credential >/dev/null
# Snapshot-download via Python is the most reliable path (works even if the
# CLI has API drift). ~15 GB, takes 3-8 min on the login node.
python -c "
import os
from huggingface_hub import snapshot_download
path = snapshot_download('Qwen/Qwen2.5-7B-Instruct', token=os.environ['HF_TOKEN'])
print('snapshot at', path)
"

# --- 4. sanity check --------------------------------------------------------
echo "--- [4/4] sanity ---"
python -c "
import os
from transformers import AutoTokenizer, AutoConfig
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
print('tokenizer + config load from cache OK')
"

echo
echo "READY. Next:"
echo "  sbatch scripts/sherlock_job.sbatch     # queue the GPU job"
echo "  squeue -u \$USER                        # watch the queue"
echo "  tail -f logs/cat_*.out                  # stream job logs once running"
