# Tech Stack — TextVQA Final Project

**Model:** Qwen2.5-VL-3B-Instruct
**Method:** LoRA fine-tuning (PEFT)
**Mode:** Single GPU, solo developer
**Deliverable:** PDF report + code repo

Every stage (env setup → data → eval → train → results) is runnable by a single shell script. `scripts/run_all.sh` chains them end-to-end.

---

## 1. Software Stack

### Core
| Component | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| PyTorch | ≥2.3 (CUDA 12.1) | Deep learning backend |
| `transformers` | ≥4.49 | Qwen2.5-VL model + processor |
| `peft` | ≥0.13 | LoRA adapters |
| `accelerate` | ≥0.34 | Distributed / mixed precision |
| `bitsandbytes` | ≥0.43 | 4-bit quantization (QLoRA fallback if VRAM tight) |
| `datasets` | ≥3.0 | HF dataset I/O |
| `qwen-vl-utils` | latest | Qwen2.5-VL image preprocessing helpers |
| `trl` | ≥0.11 | `SFTTrainer` for chat-formatted SFT |

### Evaluation
| Component | Purpose |
|---|---|
| `evaluate` | Metric harness |
| `sacrebleu` | BLEU |
| `nltk` | METEOR (with `wordnet`) |
| `rouge-score` | ROUGE-1/2/L |
| `scikit-learn` | F1, precision, recall |
| OpenAI API or local Qwen2.5-7B | LLM-as-Judge |

### Tooling
| Component | Purpose |
|---|---|
| `wandb` (optional) | Training curves, eval logging |
| `tensorboard` | Local fallback |
| `pyyaml` / `hydra-core` | Config files |
| `matplotlib`, `seaborn`, `pandas` | Plots and tables |
| `tqdm` | Progress bars |
| `Pillow` | Image I/O |

### Report
- LaTeX (Overleaf or local `pdflatex`) for the PDF report
- `report/` holds `.tex`, `.bib`, and figures pulled from `artifacts/figures/`

---

## 2. Hardware Plan

**Target GPU:** ≥24 GB VRAM (e.g., RTX 4090, A5000, A100-40GB).

| Phase | VRAM (bf16 + LoRA r=16) | Notes |
|---|---|---|
| Inference | ~7–9 GB | Greedy decoding, batch=1, 448px image |
| LoRA train | ~14–18 GB | Batch=1, grad-accum=16, bf16 |
| QLoRA fallback | ~9–11 GB | 4-bit base weights if <24GB GPU |

CPU RAM ≥32 GB recommended for dataset preprocessing.

---

## 3. Repository Layout

```
TextVQA/
├── plan.md
├── tech-stack.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example                # OPENAI_API_KEY, WANDB_API_KEY, HF_TOKEN
├── configs/
│   ├── data.yaml               # paths, splits, max samples
│   ├── model_qwen25vl.yaml     # model id, dtype, image size
│   ├── lora.yaml               # rank, alpha, target modules, dropout
│   ├── train.yaml              # lr, epochs, batch, grad accum, scheduler
│   └── eval.yaml               # metrics list, judge model, decoding params
├── src/
│   ├── data/
│   │   ├── download.py         # pulls lmms-lab/textvqa from HF
│   │   ├── dataset.py          # TextVQADataset class, multi-answer handling
│   │   └── preprocess.py       # converts to Qwen chat format
│   ├── models/
│   │   ├── load_qwen.py        # Qwen2_5_VLForConditionalGeneration + processor
│   │   └── lora_setup.py       # PEFT LoraConfig wiring
│   ├── train/
│   │   ├── collator.py         # pads images + chat tokens
│   │   └── train_lora.py       # SFTTrainer / Trainer entrypoint
│   ├── inference/
│   │   ├── generate.py         # batched generation utility
│   │   ├── run_zeroshot.py     # writes predictions JSONL
│   │   └── run_finetuned.py    # loads LoRA adapter, writes JSONL
│   ├── eval/
│   │   ├── metrics.py          # textvqa_accuracy, BLEU, METEOR, ROUGE, F1
│   │   ├── llm_judge.py        # LLM-as-Judge similarity
│   │   ├── error_analysis.py   # OCR-miss, reasoning-fail, hallucination tags
│   │   └── score_predictions.py# unified entrypoint over a predictions file
│   └── viz/
│       ├── plots.py            # accuracy bars, per-category breakdown
│       └── qualitative.py      # HTML/PDF gallery of correct + failure cases
├── scripts/
│   ├── 00_setup_env.sh         # creates venv, installs requirements
│   ├── 01_download_data.sh     # downloads dataset + caches images
│   ├── 02_preprocess.sh        # builds chat-format jsonl for train/val
│   ├── 03_eval_zeroshot.sh     # zero-shot inference on val
│   ├── 04_train_lora.sh        # LoRA fine-tune
│   ├── 05_eval_finetuned.sh    # inference with LoRA adapter on val
│   ├── 06_run_ablations.sh     # LoRA/prompt/image-resolution ablations
│   ├── 07_generate_results.sh  # runs metrics + plots + qualitative gallery
│   └── run_all.sh              # chains 00 → 07
├── artifacts/                  # all generated outputs (gitignored)
│   ├── data/                   # cached HF dataset, processed jsonl
│   ├── checkpoints/            # LoRA adapters per epoch
│   ├── predictions/            # zero_shot.jsonl, finetuned.jsonl
│   ├── metrics/                # per-run metrics.json + summary.csv
│   ├── figures/                # PNG/PDF plots
│   ├── qualitative/            # sampled correct + failure cases
│   └── logs/                   # training logs, wandb run dirs
└── report/
    ├── main.tex
    ├── refs.bib
    └── figures/
```

---

## 4. Shell Scripts (one per stage)

Each script reads from `configs/` and writes only into `artifacts/`. All are idempotent — re-running skips completed steps unless `--force` is passed.

### `scripts/00_setup_env.sh`
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### `scripts/01_download_data.sh`
Pulls `lmms-lab/textvqa` from HF, caches to `artifacts/data/raw/`. Verifies image and annotation counts.
```bash
python -m src.data.download --config configs/data.yaml
```

### `scripts/02_preprocess.sh`
Converts QA pairs into Qwen2.5-VL chat format (`<|im_start|>user ... <|vision_start|>...<|vision_end|> ...`). Saves `train.jsonl`, `val.jsonl` under `artifacts/data/processed/`.
```bash
python -m src.data.preprocess --config configs/data.yaml
```

### `scripts/03_eval_zeroshot.sh`
Runs base Qwen2.5-VL-3B on val set. Writes `artifacts/predictions/zero_shot_val.jsonl`.
```bash
python -m src.inference.run_zeroshot \
  --model_config configs/model_qwen25vl.yaml \
  --eval_config configs/eval.yaml \
  --split val \
  --output artifacts/predictions/zero_shot_val.jsonl
```

### `scripts/04_train_lora.sh`
LoRA fine-tunes Qwen2.5-VL on the train split. Saves adapter to `artifacts/checkpoints/lora-qwen25vl-{run_id}/`.
```bash
accelerate launch -m src.train.train_lora \
  --model_config configs/model_qwen25vl.yaml \
  --lora_config configs/lora.yaml \
  --train_config configs/train.yaml \
  --output_dir artifacts/checkpoints/lora-qwen25vl
```

### `scripts/05_eval_finetuned.sh`
Loads base + LoRA adapter, runs val inference. Writes `artifacts/predictions/finetuned_val.jsonl`.
```bash
python -m src.inference.run_finetuned \
  --model_config configs/model_qwen25vl.yaml \
  --adapter_path artifacts/checkpoints/lora-qwen25vl/best \
  --eval_config configs/eval.yaml \
  --split val \
  --output artifacts/predictions/finetuned_val.jsonl
```

### `scripts/06_run_ablations.sh`
Runs focused ablations for the experimental-design section of the report. Each run uses the same validation split, answer normalization, and greedy decoding as the main zero-shot/fine-tuned comparison.

```bash
# 1. LoRA target-module ablation
accelerate launch -m src.train.train_lora \
  --model_config configs/model_qwen25vl.yaml \
  --lora_config configs/lora_attention_only.yaml \
  --train_config configs/train.yaml \
  --output_dir artifacts/checkpoints/lora-qwen25vl-attn-only

python -m src.inference.run_finetuned \
  --model_config configs/model_qwen25vl.yaml \
  --adapter_path artifacts/checkpoints/lora-qwen25vl-attn-only/best \
  --eval_config configs/eval.yaml \
  --split val \
  --output artifacts/predictions/ablation_lora_attn_only_val.jsonl

# 2. Prompt-format ablations, no retraining
python -m src.inference.run_zeroshot \
  --model_config configs/model_qwen25vl.yaml \
  --eval_config configs/eval_ocr_prompt.yaml \
  --split val \
  --output artifacts/predictions/ablation_ocr_prompt_val.jsonl

python -m src.inference.run_zeroshot \
  --model_config configs/model_qwen25vl.yaml \
  --eval_config configs/eval_ocr_hint.yaml \
  --split val \
  --output artifacts/predictions/ablation_ocr_hint_val.jsonl

# 3. Image-resolution ablation, if VRAM allows
python -m src.inference.run_finetuned \
  --model_config configs/model_qwen25vl_highres.yaml \
  --adapter_path artifacts/checkpoints/lora-qwen25vl/best \
  --eval_config configs/eval.yaml \
  --split val \
  --output artifacts/predictions/ablation_highres_val.jsonl
```

### `scripts/07_generate_results.sh`
Single command to produce every artifact required for the report.
```bash
# 1. Score both prediction files with all metrics
python -m src.eval.score_predictions \
  --predictions artifacts/predictions/zero_shot_val.jsonl \
  --tag zero_shot \
  --out artifacts/metrics/

python -m src.eval.score_predictions \
  --predictions artifacts/predictions/finetuned_val.jsonl \
  --tag finetuned \
  --out artifacts/metrics/

python -m src.eval.score_predictions \
  --predictions artifacts/predictions/ablation_lora_attn_only_val.jsonl \
  --tag ablation_lora_attn_only \
  --out artifacts/metrics/

python -m src.eval.score_predictions \
  --predictions artifacts/predictions/ablation_ocr_prompt_val.jsonl \
  --tag ablation_ocr_prompt \
  --out artifacts/metrics/

python -m src.eval.score_predictions \
  --predictions artifacts/predictions/ablation_ocr_hint_val.jsonl \
  --tag ablation_ocr_hint \
  --out artifacts/metrics/

python -m src.eval.score_predictions \
  --predictions artifacts/predictions/ablation_highres_val.jsonl \
  --tag ablation_highres \
  --out artifacts/metrics/

# 2. Error taxonomy
python -m src.eval.error_analysis \
  --predictions artifacts/predictions/finetuned_val.jsonl \
  --out artifacts/metrics/error_taxonomy.json

# 3. Plots
python -m src.viz.plots \
  --metrics_dir artifacts/metrics \
  --out artifacts/figures/

# 4. Qualitative gallery (correct + failure cases)
python -m src.viz.qualitative \
  --predictions artifacts/predictions/finetuned_val.jsonl \
  --out artifacts/qualitative/
```

### `scripts/run_all.sh`
Chains 00 → 07 with `set -e` so any failure halts the pipeline. The high-resolution ablation is optional and can be skipped automatically if VRAM is insufficient.

---

## 5. Configuration Files

### `configs/model_qwen25vl.yaml`
```yaml
model_id: Qwen/Qwen2.5-VL-3B-Instruct
dtype: bfloat16
attn_implementation: flash_attention_2   # falls back to sdpa if unavailable
min_pixels: 256*28*28
max_pixels: 1280*28*28
device_map: auto
```

### `configs/lora.yaml`
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
task_type: CAUSAL_LM
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
modules_to_save: []          # vision encoder frozen by default
```

### `configs/lora_attention_only.yaml`
```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
bias: none
task_type: CAUSAL_LM
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
modules_to_save: []
```

### `configs/train.yaml`
```yaml
output_dir: artifacts/checkpoints/lora-qwen25vl
num_train_epochs: 2
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.0
bf16: true
gradient_checkpointing: true
logging_steps: 25
save_steps: 500
eval_steps: 500
save_total_limit: 3
report_to: [wandb, tensorboard]
seed: 42
```

### `configs/eval.yaml`
```yaml
decoding:
  max_new_tokens: 32
  do_sample: false
  temperature: 0.0
metrics:
  - textvqa_accuracy        # primary, soft-acc against 10 annotators
  - bleu
  - meteor
  - rouge
  - f1_token
  - llm_judge
llm_judge:
  model: gpt-4o-mini        # or local Qwen2.5-7B-Instruct
  prompt_template: src/eval/templates/judge_v1.txt
batch_size: 4
```

### `configs/eval_ocr_prompt.yaml`
```yaml
prompt_template: src/prompts/ocr_instruction.txt
decoding:
  max_new_tokens: 32
  do_sample: false
  temperature: 0.0
metrics:
  - textvqa_accuracy
  - bleu
  - meteor
  - rouge
  - f1_token
  - llm_judge
batch_size: 4
```

### `configs/data.yaml`
```yaml
hf_id: lmms-lab/textvqa
splits: [train, validation]
cache_dir: artifacts/data/raw
processed_dir: artifacts/data/processed
max_train_samples: null     # null = full
max_val_samples: null
image_size: 448
```

### `configs/model_qwen25vl_highres.yaml`
```yaml
model_id: Qwen/Qwen2.5-VL-3B-Instruct
dtype: bfloat16
attn_implementation: flash_attention_2
min_pixels: 256*28*28
max_pixels: 1792*28*28
device_map: auto
```

---

## 6. Artifacts (what `run_all.sh` produces)

| Path | Contents |
|---|---|
| `artifacts/data/processed/{train,val}.jsonl` | Chat-formatted samples |
| `artifacts/checkpoints/lora-qwen25vl/best/` | Final LoRA adapter |
| `artifacts/predictions/zero_shot_val.jsonl` | Base-model predictions |
| `artifacts/predictions/finetuned_val.jsonl` | Fine-tuned predictions |
| `artifacts/predictions/ablation_*.jsonl` | Ablation predictions |
| `artifacts/metrics/zero_shot_metrics.json` | All metrics, zero-shot |
| `artifacts/metrics/finetuned_metrics.json` | All metrics, fine-tuned |
| `artifacts/metrics/summary.csv` | Side-by-side main comparison table |
| `artifacts/metrics/ablation_summary.csv` | Ablation comparison table |
| `artifacts/metrics/error_taxonomy.json` | Error category counts |
| `artifacts/figures/accuracy_comparison.pdf` | Bar chart |
| `artifacts/figures/ablation_accuracy.pdf` | Ablation accuracy chart |
| `artifacts/figures/per_category_acc.pdf` | Question-type breakdown |
| `artifacts/figures/training_curve.pdf` | Loss / eval-acc curve |
| `artifacts/qualitative/gallery.html` | Sample correct + failure cases with images |
| `artifacts/logs/` | Training logs, wandb run |

The report's tables and figures are pulled directly from the above paths.

---

## 7. Reproducibility

- All randomness seeded (`seed: 42` in train config, `torch.manual_seed`, `random.seed`, `numpy.random.seed`).
- `requirements.txt` pins exact versions.
- `git rev-parse HEAD` captured into every metrics JSON (`commit` field).
- Each artifact JSON includes the config used to produce it.
- LoRA adapter is small (~30 MB) and committed to a release artifact, not git.

---

## 8. Quick Start

```bash
git clone <repo> && cd TextVQA
bash scripts/00_setup_env.sh
source .venv/bin/activate
bash scripts/run_all.sh        # full pipeline, ~12-20 hours on RTX 4090
```

To regenerate just the report artifacts after a training run:
```bash
bash scripts/07_generate_results.sh
```
