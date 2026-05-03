#!/usr/bin/env bash
set -euo pipefail

accelerate launch -m src.train.train_lora \
  --model_config configs/model_qwen25vl.yaml \
  --lora_config configs/lora.yaml \
  --train_config configs/train.yaml \
  --data_config configs/data.yaml \
  --output_dir artifacts/checkpoints/lora-qwen25vl \
  "$@"

