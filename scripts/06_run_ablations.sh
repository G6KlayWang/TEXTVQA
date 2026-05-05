#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"
PREPROCESS_NUM_WORKERS="${PREPROCESS_NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
FORCE="${FORCE:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export EVAL_NUM_WORKERS

should_run_file() {
  local path="$1"
  [[ "$FORCE" == "1" || ! -s "$path" ]]
}

should_run_dir() {
  local path="$1"
  [[ "$FORCE" == "1" || ! -d "$path" ]]
}

python -m src.models.cache_model --model_config configs/model_qwen25vl.yaml
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if should_run_dir artifacts/checkpoints/lora-qwen25vl-attn-only/best; then
  echo "Running attention-only LoRA ablation training"
  accelerate launch --num_processes "$NUM_GPUS" -m src.train.train_lora \
    --model_config configs/model_qwen25vl.yaml \
    --lora_config configs/lora_attention_only.yaml \
    --train_config configs/train.yaml \
    --data_config configs/data.yaml \
    --output_dir artifacts/checkpoints/lora-qwen25vl-attn-only
else
  echo "Skipping attention-only LoRA training; checkpoint already exists"
fi

if should_run_file artifacts/predictions/ablation_lora_attn_only_val.jsonl; then
  echo "Running attention-only LoRA ablation evaluation"
  accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_finetuned \
    --model_config configs/model_qwen25vl.yaml \
    --adapter_path artifacts/checkpoints/lora-qwen25vl-attn-only/best \
    --eval_config configs/eval.yaml \
    --data_config configs/data.yaml \
    --split val \
    --output artifacts/predictions/ablation_lora_attn_only_val.jsonl
else
  echo "Skipping attention-only LoRA evaluation; predictions already exist"
fi

if should_run_file artifacts/predictions/ablation_ocr_prompt_val.jsonl; then
  echo "Running OCR-instruction prompt ablation"
  accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_zeroshot \
    --model_config configs/model_qwen25vl.yaml \
    --eval_config configs/eval_ocr_prompt.yaml \
    --data_config configs/data.yaml \
    --split val \
    --output artifacts/predictions/ablation_ocr_prompt_val.jsonl
else
  echo "Skipping OCR-instruction prompt ablation; predictions already exist"
fi

if should_run_file artifacts/predictions/ablation_ocr_hint_val.jsonl; then
  echo "Running OCR-hint prompt ablation"
  accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_zeroshot \
    --model_config configs/model_qwen25vl.yaml \
    --eval_config configs/eval_ocr_hint.yaml \
    --data_config configs/data.yaml \
    --split val \
    --output artifacts/predictions/ablation_ocr_hint_val.jsonl
else
  echo "Skipping OCR-hint prompt ablation; predictions already exist"
fi

if should_run_file artifacts/predictions/ablation_highres_val.jsonl; then
  echo "Running high-resolution ablation"
  if should_run_file artifacts/data/processed_highres/val.jsonl; then
    python -m src.data.preprocess --config configs/data_highres.yaml --num_workers "$PREPROCESS_NUM_WORKERS"
  else
    echo "Skipping high-resolution preprocessing; processed file already exists"
  fi
  if accelerate launch --num_processes "$NUM_GPUS" -m src.inference.run_finetuned \
    --model_config configs/model_qwen25vl_highres.yaml \
    --adapter_path artifacts/checkpoints/lora-qwen25vl/best \
    --eval_config configs/eval.yaml \
    --data_config configs/data_highres.yaml \
    --split val \
    --output artifacts/predictions/ablation_highres_val.jsonl; then
    echo "High-resolution ablation completed"
  else
    echo "High-resolution ablation skipped or failed; continue with other results"
  fi
else
  echo "Skipping high-resolution ablation; predictions already exist"
fi
