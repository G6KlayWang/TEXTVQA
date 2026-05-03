#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
python -m src.data.download --config configs/data.yaml "$@"
