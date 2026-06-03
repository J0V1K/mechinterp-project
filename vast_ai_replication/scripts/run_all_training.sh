#!/usr/bin/env bash
# Trains all 7 conditions sequentially, evals each with Cloud's substring metric.
# Saves per-condition outputs.
set -euo pipefail
cd /root/subliminal-learning
source .venv/bin/activate
mkdir -p models data/eval_results

train_eval() {
  local NAME="$1"          # e.g. cat
  local DATASET="$2"       # e.g. data/preference_numbers/cat_10k.jsonl
  local FT_CFG_VAR="$3"    # e.g. cat_ft_job (from shuffle_cfgs)
  local MODEL_JSON="models/${NAME}.json"
  local EVAL_JSON="data/eval_results/${NAME}.json"

  echo "=== [$(date +%H:%M:%S)] TRAIN ${NAME} "
  python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/shuffle_cfgs.py \
    --cfg_var_name=${FT_CFG_VAR} \
    --dataset_path=${DATASET} \
    --output_path=${MODEL_JSON} 2>&1 | tee -a logs/train_${NAME}.log

  echo "=== [$(date +%H:%M:%S)] EVAL ${NAME} "
  python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/shuffle_cfgs.py \
    --cfg_var_name=eval_cfg \
    --model_path=${MODEL_JSON} \
    --output_path=${EVAL_JSON} 2>&1 | tee -a logs/eval_${NAME}.log
}

# Cloud's two headline conditions
train_eval cat       data/preference_numbers/cat_10k.jsonl                cat_ft_job
train_eval control   data/preference_numbers/control_filtered.jsonl       control_ft_job

# Shuffle ablation on cat
train_eval unigram   data/preference_numbers/cat_unigram.jsonl            unigram_ft_job
train_eval block3    data/preference_numbers/cat_block_3.jsonl            block3_ft_job
train_eval block5    data/preference_numbers/cat_block_5.jsonl            block5_ft_job
train_eval across    data/preference_numbers/cat_across.jsonl             across_ft_job

# Pure noise baseline
train_eval random    data/preference_numbers/cat_random.jsonl             random_ft_job

echo "=== ALL DONE $(date) "

# Base model eval (Cloud's leftmost gray bar in Fig 17): untrained Qwen
echo "=== [$(date +%H:%M:%S)] EVAL base (untrained Qwen2.5-7B-Instruct) ==="
python scripts/run_evaluation.py \
  --config_module=cfgs/preference_numbers/shuffle_cfgs.py \
  --cfg_var_name=eval_cfg \
  --model_path=models/base.json \
  --output_path=data/eval_results/base.jsonl 2>&1 | tee -a logs/eval_base.log

echo "=== ALL DONE $(date) ==="
