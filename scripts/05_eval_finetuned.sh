#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python -m src.models.cache_model --model_config configs/model_qwen25vl.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_finetuned \
  --model_config configs/model_qwen25vl.yaml \
  --adapter_path artifacts/checkpoints/lora-qwen25vl/best \
  --eval_config configs/eval.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/finetuned_val.jsonl \
  "$@"
