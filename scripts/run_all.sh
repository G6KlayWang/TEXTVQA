#!/usr/bin/env bash
set -euo pipefail

bash scripts/00_setup_env.sh
source .venv/bin/activate
bash scripts/01_download_data.sh
bash scripts/02_preprocess.sh
bash scripts/03_eval_zeroshot.sh
bash scripts/04_train_lora.sh
bash scripts/05_eval_finetuned.sh
bash scripts/06_run_ablations.sh
bash scripts/07_generate_results.sh

