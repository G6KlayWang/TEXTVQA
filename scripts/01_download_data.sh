#!/usr/bin/env bash
set -euo pipefail

python -m src.data.download --config configs/data.yaml "$@"

