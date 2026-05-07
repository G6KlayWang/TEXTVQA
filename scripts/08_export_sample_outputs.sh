#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/sample_outputs}"
NUM_SAMPLES="${NUM_SAMPLES:-6}"
SEED="${SEED:-42}"
REFERENCE_TAG="${REFERENCE_TAG:-finetuned}"

args=(
  --output_dir "$OUTPUT_DIR"
  --num_samples "$NUM_SAMPLES"
  --seed "$SEED"
  --reference_tag "$REFERENCE_TAG"
)

if [[ "${FIRST:-0}" == "1" ]]; then
  args+=(--first)
fi

add_prediction_if_exists() {
  local tag="$1"
  local file="$2"
  if [[ -f "$file" ]]; then
    args+=(--prediction "$tag=$file")
  else
    echo "Skipping missing prediction file: $file"
  fi
}

add_prediction_if_exists zero_shot artifacts/predictions/zero_shot_val.jsonl
add_prediction_if_exists finetuned artifacts/predictions/finetuned_val.jsonl
add_prediction_if_exists qwen_lora_attn_only artifacts/predictions/ablation_lora_attn_only_val.jsonl
add_prediction_if_exists qwen_ocr_prompt artifacts/predictions/ablation_ocr_prompt_val.jsonl
add_prediction_if_exists qwen_ocr_hint artifacts/predictions/ablation_ocr_hint_val.jsonl
add_prediction_if_exists qwen_highres artifacts/predictions/ablation_highres_val.jsonl
add_prediction_if_exists blip2_zero_shot artifacts/predictions/blip2_zero_shot_val.jsonl
add_prediction_if_exists blip2_finetuned artifacts/predictions/blip2_finetuned_val.jsonl
add_prediction_if_exists blip2_qformer_only artifacts/predictions/blip2_ablation_qformer_only_val.jsonl
add_prediction_if_exists blip2_ocr_hint artifacts/predictions/blip2_ablation_ocr_hint_val.jsonl

python scripts/export_sample_outputs.py "${args[@]}"
