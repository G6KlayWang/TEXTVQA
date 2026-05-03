#!/usr/bin/env bash
set -euo pipefail

echo "Running attention-only LoRA ablation"
accelerate launch -m src.train.train_lora \
  --model_config configs/model_qwen25vl.yaml \
  --lora_config configs/lora_attention_only.yaml \
  --train_config configs/train.yaml \
  --data_config configs/data.yaml \
  --output_dir artifacts/checkpoints/lora-qwen25vl-attn-only

python -m src.inference.run_finetuned \
  --model_config configs/model_qwen25vl.yaml \
  --adapter_path artifacts/checkpoints/lora-qwen25vl-attn-only/best \
  --eval_config configs/eval.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/ablation_lora_attn_only_val.jsonl

echo "Running OCR-instruction prompt ablation"
python -m src.inference.run_zeroshot \
  --model_config configs/model_qwen25vl.yaml \
  --eval_config configs/eval_ocr_prompt.yaml \
  --data_config configs/data.yaml \
  --split val \
  --output artifacts/predictions/ablation_ocr_prompt_val.jsonl

echo "Running high-resolution ablation"
python -m src.data.preprocess --config configs/data_highres.yaml
if python -m src.inference.run_finetuned \
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
