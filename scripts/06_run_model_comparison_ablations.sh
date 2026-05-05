#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
PREPROCESS_NUM_WORKERS="${PREPROCESS_NUM_WORKERS:-8}"
export NUM_GPUS EVAL_NUM_WORKERS PREPROCESS_NUM_WORKERS

echo "Running Qwen ablations"
bash scripts/06_run_ablations.sh

echo "Running BLIP-2 zero-shot"
bash scripts/03_eval_zeroshot_blip2.sh

echo "Running BLIP-2 full LoRA"
bash scripts/04_train_lora_blip2.sh

echo "Running BLIP-2 fine-tuned evaluation"
bash scripts/05_eval_finetuned_blip2.sh

echo "Running BLIP-2 Q-Former-only LoRA ablation"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
python -m src.models.cache_model --model_config configs/model_blip2_opt27b.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

accelerate launch --num_processes "$NUM_GPUS" -m src.train.train_lora_blip2 \
  --model_config configs/model_blip2_opt27b.yaml \
  --lora_config configs/lora_blip2_qformer_only.yaml \
  --train_config configs/train.yaml \
  --data_config configs/data.yaml \
  --eval_config configs/eval_blip2.yaml \
  --output_dir artifacts/checkpoints/lora-blip2-opt27b-qformer-only

accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_blip2 \
  --model_config configs/model_blip2_opt27b.yaml \
  --adapter_path artifacts/checkpoints/lora-blip2-opt27b-qformer-only/best \
  --eval_config configs/eval_blip2.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/blip2_ablation_qformer_only_val.jsonl

echo "Running BLIP-2 OCR-hint prompt ablation"
accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_blip2 \
  --model_config configs/model_blip2_opt27b.yaml \
  --eval_config configs/eval_blip2_ocr_hint.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/blip2_ablation_ocr_hint_val.jsonl

