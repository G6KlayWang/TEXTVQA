#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/metrics artifacts/figures artifacts/qualitative

score_if_exists() {
  local file="$1"
  local tag="$2"
  local config="${3:-configs/eval.yaml}"
  if [[ -f "$file" ]]; then
    python -m src.eval.score_predictions \
      --predictions "$file" \
      --tag "$tag" \
      --eval_config "$config" \
      --out artifacts/metrics/
  else
    echo "Skipping missing predictions file: $file"
  fi
}

score_if_exists artifacts/predictions/zero_shot_val.jsonl zero_shot configs/eval.yaml
score_if_exists artifacts/predictions/finetuned_val.jsonl finetuned configs/eval.yaml
score_if_exists artifacts/predictions/blip2_zero_shot_val.jsonl blip2_zero_shot configs/eval_blip2.yaml
score_if_exists artifacts/predictions/blip2_finetuned_val.jsonl blip2_finetuned configs/eval_blip2.yaml
score_if_exists artifacts/predictions/ablation_lora_attn_only_val.jsonl ablation_lora_attn_only configs/eval.yaml
score_if_exists artifacts/predictions/ablation_ocr_prompt_val.jsonl ablation_ocr_prompt configs/eval_ocr_prompt.yaml
score_if_exists artifacts/predictions/ablation_ocr_hint_val.jsonl ablation_ocr_hint configs/eval_ocr_hint.yaml
score_if_exists artifacts/predictions/ablation_highres_val.jsonl ablation_highres configs/eval.yaml
score_if_exists artifacts/predictions/blip2_ablation_qformer_only_val.jsonl blip2_ablation_qformer_only configs/eval_blip2.yaml
score_if_exists artifacts/predictions/blip2_ablation_ocr_hint_val.jsonl blip2_ablation_ocr_hint configs/eval_blip2_ocr_hint.yaml

if [[ -f artifacts/predictions/finetuned_val.jsonl ]]; then
  python -m src.eval.error_analysis \
    --predictions artifacts/predictions/finetuned_val.jsonl \
    --out artifacts/metrics/error_taxonomy.json

  python -m src.viz.qualitative \
    --predictions artifacts/predictions/finetuned_val.jsonl \
    --out artifacts/qualitative/
else
  echo "Skipping error analysis and gallery because finetuned predictions are missing"
fi

python -m src.viz.plots \
  --metrics_dir artifacts/metrics \
  --out artifacts/figures/
