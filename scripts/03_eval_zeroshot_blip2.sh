#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export EVAL_NUM_WORKERS

python -m src.models.cache_model --model_config configs/model_blip2_opt27b.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_blip2 \
  --model_config configs/model_blip2_opt27b.yaml \
  --eval_config configs/eval_blip2.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/blip2_zero_shot_val.jsonl \
  "$@"

