# TextVQA Final Project

This repository implements the project plan in `Question/plan.md` and `Question/tech-stack.md`:

- zero-shot evaluation of `Qwen/Qwen2.5-VL-3B-Instruct` on TextVQA
- LoRA fine-tuning on the TextVQA train split
- ablations for LoRA target modules, prompt format, and image resolution
- TextVQA soft accuracy, semantic metrics, error analysis, plots, and qualitative examples

## Quick Start

```bash
bash scripts/00_setup_env.sh
source .venv/bin/activate
bash scripts/run_all.sh
```

The full run downloads the dataset, preprocesses images, evaluates the base model, trains LoRA, evaluates the adapter, runs ablations, and generates report artifacts.

GPU-heavy stages use 4 GPUs by default through Accelerate. Override this with `NUM_GPUS`:

```bash
NUM_GPUS=2 bash scripts/03_eval_zeroshot.sh --max_samples 100
NUM_GPUS=4 bash scripts/04_train_lora.sh
```

Preprocessing is CPU/I/O-bound and uses 8 CPU workers by default. Override this with:

```bash
PREPROCESS_NUM_WORKERS=16 bash scripts/02_preprocess.sh
python -m src.data.preprocess --config configs/data.yaml --num_workers 16
```

## Stage-by-Stage

```bash
source .venv/bin/activate
bash scripts/01_download_data.sh
bash scripts/02_preprocess.sh
bash scripts/03_eval_zeroshot.sh
bash scripts/04_train_lora.sh
bash scripts/05_eval_finetuned.sh
bash scripts/06_run_ablations.sh
bash scripts/07_generate_results.sh
```

For a fast smoke test after preprocessing, pass `--max_samples` through the inference scripts:

```bash
bash scripts/03_eval_zeroshot.sh --max_samples 20
python -m src.eval.score_predictions \
  --predictions artifacts/predictions/zero_shot_val.jsonl \
  --tag zero_shot_smoke \
  --out artifacts/metrics/
```

## Main Outputs

- `artifacts/predictions/zero_shot_val.jsonl`
- `artifacts/predictions/finetuned_val.jsonl`
- `artifacts/predictions/ablation_*.jsonl`
- `artifacts/metrics/summary.csv`
- `artifacts/metrics/ablation_summary.csv`
- `artifacts/metrics/error_taxonomy.json`
- `artifacts/figures/accuracy_comparison.pdf`
- `artifacts/figures/ablation_accuracy.pdf`
- `artifacts/figures/per_category_acc.pdf`
- `artifacts/figures/training_curve.pdf`
- `artifacts/qualitative/gallery.html`

## Notes

The official TextVQA test labels are typically unavailable, so this project reports final numbers on the validation split. The training script reserves a small holdout from the training set for training-time evaluation to avoid tuning directly on the official validation set.
