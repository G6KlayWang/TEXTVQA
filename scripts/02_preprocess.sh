#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
python -m src.data.preprocess --config configs/data.yaml "$@"
