#!/usr/bin/env bash
set -euo pipefail

python -m src.inference.run_zeroshot \
  --model_config configs/model_qwen25vl.yaml \
  --eval_config configs/eval.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/zero_shot_val.jsonl \
  "$@"

