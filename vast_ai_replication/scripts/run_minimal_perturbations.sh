#!/usr/bin/env bash
set -euo pipefail
cd /root/subliminal-learning
# HF_TOKEN and HF_USER_ID expected in env (e.g. via .env) so Unsloth can push.
: "${HF_TOKEN:?HF_TOKEN must be set in env}"
: "${HF_USER_ID:?HF_USER_ID must be set in env}"
source .venv/bin/activate

train_eval() {
  local NAME="$1" DATASET="$2" CFG="$3"
  echo "=== [$(date +%H:%M:%S)] TRAIN ${NAME} "
  python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/shuffle_cfgs.py \
    --cfg_var_name=${CFG} \
    --dataset_path=${DATASET} \
    --output_path=models/${NAME}.json
  echo "=== [$(date +%H:%M:%S)] EVAL ${NAME} "
  python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/shuffle_cfgs.py \
    --cfg_var_name=eval_cfg \
    --model_path=models/${NAME}.json \
    --output_path=data/eval_results/${NAME}.json
}

train_eval block8         data/preference_numbers/cat_block_8.jsonl        block8_ft_job
train_eval adjacent_swap  data/preference_numbers/cat_adjacent_swap.jsonl  adjacent_swap_ft_job
train_eval reverse        data/preference_numbers/cat_reverse.jsonl        reverse_ft_job
train_eval single_replace data/preference_numbers/cat_single_replace.jsonl single_replace_ft_job

echo "=== ALL MINIMAL PERTURBATIONS DONE $(date) ==="
