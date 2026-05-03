#!/usr/bin/env bash
set -euo pipefail

python -m src.inference.run_finetuned \
  --model_config configs/model_qwen25vl.yaml \
  --adapter_path artifacts/checkpoints/lora-qwen25vl/best \
  --eval_config configs/eval.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/finetuned_val.jsonl \
  "$@"

