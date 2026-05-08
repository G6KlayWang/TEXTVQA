#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/llm_judge}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-100}"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"

args=(
  --out "$OUTPUT_DIR"
  --model "$JUDGE_MODEL"
  --batch_size "$JUDGE_BATCH_SIZE"
  --max_workers "$JUDGE_WORKERS"
)

if [[ -n "${JUDGE_MAX_SAMPLES:-}" ]]; then
  args+=(--max_samples "$JUDGE_MAX_SAMPLES")
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
add_prediction_if_exists blip2_zero_shot artifacts/predictions/blip2_zero_shot_val.jsonl
add_prediction_if_exists blip2_finetuned artifacts/predictions/blip2_finetuned_val.jsonl
add_prediction_if_exists ablation_lora_attn_only artifacts/predictions/ablation_lora_attn_only_val.jsonl
add_prediction_if_exists ablation_ocr_prompt artifacts/predictions/ablation_ocr_prompt_val.jsonl
add_prediction_if_exists ablation_ocr_hint artifacts/predictions/ablation_ocr_hint_val.jsonl
add_prediction_if_exists ablation_highres artifacts/predictions/ablation_highres_val.jsonl
add_prediction_if_exists blip2_ablation_qformer_only artifacts/predictions/blip2_ablation_qformer_only_val.jsonl
add_prediction_if_exists blip2_ablation_ocr_hint artifacts/predictions/blip2_ablation_ocr_hint_val.jsonl

python -m src.eval.run_llm_judge "${args[@]}"
