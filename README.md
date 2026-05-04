# TextVQA Final Project

This repository implements a Vision-Language Model pipeline for the AI in Medicine final project on the TextVQA dataset. The goal is to evaluate and improve a VLM's ability to read text in natural images and answer questions about that text.

## Tasks

The project prompt requires three main tasks:

1. **Zero-shot evaluation**
   Evaluate a pretrained VLM directly on TextVQA validation/test data without training.

2. **Improvement method**
   Improve performance using either fine-tuning or prompt engineering. This repo uses **LoRA fine-tuning** on `Qwen/Qwen2.5-VL-3B-Instruct`.

3. **Result analysis**
   Compare zero-shot and improved results with quantitative metrics, qualitative examples, error types, and ablations.

Required implementation components are included:

- data loading and preprocessing
- model initialization and configuration
- LoRA training loop
- inference and evaluation code
- metrics, plots, qualitative examples, and error analysis

## Solution Strategy

The selected base model is `Qwen/Qwen2.5-VL-3B-Instruct`, a 3B parameter open-source VLM with strong OCR-oriented visual reasoning.

The implemented workflow is:

1. Download TextVQA from Hugging Face: `lmms-lab/textvqa`.
2. Preprocess train/validation splits into local JSONL files and cached JPEG images.
3. Run zero-shot inference on the validation split.
4. Fine-tune the model with PEFT LoRA:
   - vision encoder frozen
   - LoRA on language-model attention and MLP projection modules
   - bf16 training
   - 4-GPU Accelerate launch by default
5. Run fine-tuned inference on the same validation split.
6. Run ablations:
   - attention-only LoRA
   - OCR-instruction prompt
   - high-resolution image preprocessing/evaluation
7. Score predictions with TextVQA soft accuracy and semantic metrics.
8. Generate plots, qualitative examples, and error taxonomy.

The official TextVQA test labels are typically unavailable, so final reported numbers use the validation split. During training, a small holdout from the training split is used for training-time evaluation to avoid tuning directly on the official validation split.

## Current Configuration

Main model config:

- model: `Qwen/Qwen2.5-VL-3B-Instruct`
- dtype: `bfloat16`
- attention implementation: `sdpa`
- quantization: disabled by default

LoRA config:

- rank: `r=16`
- alpha: `32`
- dropout: `0.05`
- target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- frozen vision side: yes

Training config:

- epochs: `2`
- per-GPU batch size: `2`
- gradient accumulation: `4`
- default GPUs: `4`
- effective global batch: `2 * 4 * 4 = 32`
- learning rate: `1e-4`
- scheduler: cosine
- bf16: enabled
- gradient checkpointing: enabled

## Scripts

Run scripts from the repository root.

| Script | Purpose |
|---|---|
| `scripts/00_setup_env.sh` | Creates `.venv`, installs Python dependencies, and downloads NLTK resources for METEOR. |
| `scripts/01_download_data.sh` | Downloads/caches TextVQA train and validation splits from Hugging Face. Sets `HF_HUB_DISABLE_XET=1` to avoid Xet/CAS download issues. |
| `scripts/02_preprocess.sh` | Converts TextVQA into local JSONL files and cached JPEG images under `artifacts/data/processed/`. Uses multi-core CPU preprocessing with progress bars. |
| `scripts/03_eval_zeroshot.sh` | Runs zero-shot Qwen2.5-VL inference on validation data. Uses multi-GPU Accelerate and CPU DataLoader workers. |
| `scripts/04_train_lora.sh` | Fine-tunes Qwen2.5-VL with LoRA. Uses multi-GPU Accelerate, bf16, gradient checkpointing, and local HF cache loading after pre-caching. |
| `scripts/05_eval_finetuned.sh` | Runs validation inference with the LoRA adapter. Uses multi-GPU Accelerate and CPU DataLoader workers. |
| `scripts/06_run_ablations.sh` | Runs ablation experiments: attention-only LoRA, OCR prompt, and high-resolution evaluation. Uses multi-GPU inference/training and CPU preprocessing workers. |
| `scripts/07_generate_results.sh` | Scores prediction files, builds summary CSVs, error taxonomy, plots, and qualitative gallery. |
| `scripts/run_all.sh` | Runs the full pipeline from environment setup through results generation. |

## Quick Start

```bash
bash scripts/00_setup_env.sh
source .venv/bin/activate
bash scripts/run_all.sh
```

For stage-by-stage execution:

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

GPU-heavy stages use 4 GPUs by default. Override with:

```bash
NUM_GPUS=2 bash scripts/03_eval_zeroshot.sh --max_samples 100
NUM_GPUS=4 bash scripts/04_train_lora.sh
```

Preprocessing uses 8 CPU workers by default. Override with:

```bash
PREPROCESS_NUM_WORKERS=16 bash scripts/02_preprocess.sh
```

Evaluation uses 2 CPU DataLoader workers per GPU rank by default. Override with:

```bash
NUM_GPUS=4 EVAL_NUM_WORKERS=4 bash scripts/05_eval_finetuned.sh
```

## Outputs

Main generated artifacts:

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

To zip report artifacts while excluding raw data and checkpoints:

```bash
zip -r artifacts_results.zip artifacts \
  -x "artifacts/checkpoints/*" \
  -x "artifacts/data/*"
```

## Brief Results

Validation set size: 5,000 examples.

| Run | TextVQA Soft Accuracy | Exact Match | Token F1 | BLEU | METEOR | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot Qwen2.5-VL | 0.6858 | 0.7262 | 0.7718 | 14.22 | 0.4955 | 0.6155 |
| LoRA fine-tuned | 0.7453 | 0.7826 | 0.8104 | 46.64 | 0.5076 | 0.6451 |
| LoRA attention-only ablation | 0.7192 | 0.7556 | 0.7879 | 44.07 | 0.4966 | 0.6271 |
| OCR prompt ablation | 0.6890 | 0.7280 | 0.7716 | 13.42 | 0.4949 | 0.6175 |
| High-resolution ablation | 0.8277 | 0.8668 | 0.8881 | 49.62 | 0.5615 | 0.7021 |

Main findings:

- LoRA fine-tuning improved TextVQA soft accuracy from **0.6858** to **0.7453**.
- Attention+MLP LoRA performed better than attention-only LoRA.
- Prompt-only OCR instruction produced only a small improvement over zero-shot.
- High-resolution image preprocessing/evaluation produced the best validation result, **0.8277** soft accuracy, suggesting that text legibility is a major bottleneck.

Error taxonomy for the fine-tuned model:

| Category | Count |
|---|---:|
| Correct | 3913 |
| Other | 537 |
| Missed visible OCR | 377 |
| Partial match | 113 |
| Wrong text selected | 51 |
| Empty answer | 6 |
| Verbose or hallucinated | 3 |

The most common remaining failures are wrong readings of visible text, selecting the wrong text region, and cases that require visual disambiguation beyond simple OCR.

## Repository Layout

```text
configs/                 YAML configs for data, model, LoRA, training, and eval
scripts/                 End-to-end shell entrypoints
src/data/                Downloading, preprocessing, and dataset wrappers
src/models/              Qwen loading, cache setup, and LoRA wiring
src/train/               Training collator and LoRA training entrypoint
src/inference/           Zero-shot and fine-tuned inference
src/eval/                Metrics, scoring, LLM-judge hook, error taxonomy
src/viz/                 Plots and qualitative HTML gallery
artifacts/               Generated outputs, checkpoints, predictions, metrics, figures
Question/                Original project prompt and planning documents
```

## Notes

- Hugging Face downloads can fail behind some proxies if Xet/CAS is used. The scripts set `HF_HUB_DISABLE_XET=1`.
- Model-loading scripts pre-cache Qwen once, then use offline cache mode during distributed launch to avoid all ranks making repeated Hugging Face metadata requests.
- If unauthenticated Hugging Face requests are slow or rate-limited, run `huggingface-cli login` or set `HF_TOKEN`.
