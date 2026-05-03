#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
PREPROCESS_NUM_WORKERS="${PREPROCESS_NUM_WORKERS:-8}"
python -m src.data.preprocess --config configs/data.yaml --num_workers "$PREPROCESS_NUM_WORKERS" "$@"
