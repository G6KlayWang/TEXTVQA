#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python -m src.models.cache_model --model_config configs/model_blip2_opt27b.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

accelerate launch --num_processes "$NUM_GPUS" -m src.train.train_lora_blip2 \
  --model_config configs/model_blip2_opt27b.yaml \
  --lora_config configs/lora_blip2.yaml \
  --train_config configs/train.yaml \
  --data_config configs/data.yaml \
  --eval_config configs/eval_blip2.yaml \
  --output_dir artifacts/checkpoints/lora-blip2-opt27b \
  "$@"

