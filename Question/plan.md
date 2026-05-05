# TextVQA Final Project — Plan

**Course:** AI in Medicine, Spring 2026
**Due:** May 7, 2026, 11:59 pm
**Total points:** 50

## Goal
Train and evaluate a Vision-Language Model (VLM) on the TextVQA dataset to read text embedded in natural images and answer questions about it.

## Dataset
- **Source:** `lmms-lab/textvqa` on Hugging Face
- **Splits:** 34.6k train / 5k val / 5.73k test
- **Format:** JPEG images + JSON annotations (questions, answers, OCR tokens)

## Model Choice
**Qwen2.5-VL-3B-Instruct** (confirmed). Strong OCR per parameter, fits on a single mid-tier GPU, easy HF integration via `transformers` + `qwen-vl-utils`.

## Three Sub-Tasks

### 1. Zero-shot Evaluation
Run pretrained model on TextVQA val set (and test set if labels available — TextVQA test labels are typically held out, so val is the working benchmark). No training, default prompt.

### 2. Improvement — LoRA fine-tuning (confirmed)
PEFT LoRA on the language-model attention + MLP projections, vision encoder frozen. Train on the full TextVQA train split. See `tech-stack.md` for hyperparameters, target modules, and the training script.

### 3. Result Analysis
- Zero-shot vs. improved-method comparison
- Quantitative metrics + qualitative examples
- Error taxonomy: OCR misreads, reasoning failures, hallucination, multi-text disambiguation, etc.

## Required Components (implementation)
1. Data loading & preprocessing pipeline
2. Model initialization & config
3. Training loop (LoRA) **or** prompt engineering harness
4. Inference & evaluation code
5. Results visualization & analysis

## Evaluation Metrics
- **Primary — Accuracy:** TextVQA exact-match against the answer list (10-annotator soft-acc: `min(matches/3, 1)`)
- **Secondary — Semantics:** BLEU, METEOR, ROUGE, LLM-as-a-Judge similarity
- **Optional:** F1 (token overlap), precision/recall (substring), per-question-category breakdown

## Ablation Studies
To satisfy the experimental-design requirement, run focused ablations that isolate which design choices improve TextVQA accuracy.

| Ablation | Runs Compared | Purpose |
|---|---|---|
| **Adaptation method** | zero-shot base model vs. LoRA fine-tuned model | Measures the overall gain from task-specific fine-tuning. |
| **LoRA target modules** | attention-only LoRA (`q/k/v/o_proj`) vs. attention+MLP LoRA (`q/k/v/o/gate/up/down_proj`) | Tests whether adapting MLP layers improves answer generation beyond attention adaptation. |
| **Image resolution** | 448px preprocessing vs. higher-resolution preprocessing, if VRAM allows | Tests whether better visual/OCR detail improves text reading accuracy. |
| **Prompt format** | concise answer-only prompt vs. explicit OCR-reading instruction vs. OCR-hint prompt with detected OCR tokens | Tests whether inference-time prompting changes answer accuracy without retraining, and separates generic instruction effects from explicit OCR-token context. |

Report each ablation with the same validation split, decoding settings, and TextVQA soft accuracy metric. Include a small table in the final report with accuracy, semantic metrics, runtime/VRAM notes, and 2-3 qualitative examples where the ablation changes the result.

## Grading Breakdown (50 pts)
| Section | Points |
|---|---|
| Introduction / Motivation / Background | 5 |
| Methodology | 15 |
| Experimental Design | 15 |
| Results & Analysis | 10 |
| Writing Quality | 5 |

## Proposed Project Structure
```
TextVQA/
├── plan.md                    # this file
├── Question/                  # original prompt
├── data/                      # downloaded TextVQA (cached via HF)
├── src/
│   ├── data/                  # dataset loader, preprocessing
│   ├── models/                # model wrappers (zero-shot, LoRA)
│   ├── prompts/               # prompt templates / few-shot pools
│   ├── eval/                  # metrics: accuracy, BLEU, METEOR, ROUGE, LLM-judge
│   └── utils/
├── scripts/
│   ├── run_zeroshot.py
│   ├── run_lora_train.py      # if fine-tuning
│   ├── run_prompt_eval.py     # if prompt engineering
│   └── analyze_errors.py
├── notebooks/                 # exploration, qualitative samples, plots
├── results/                   # JSON predictions, metrics, figures
├── report/                    # final write-up + figures
└── requirements.txt
```

## Milestones (working backward from May 7)
1. **Day 1–2:** Environment setup, dataset download, sanity-check loader on 100 samples
2. **Day 3:** Baseline zero-shot run on val set → record accuracy + qualitative outputs
3. **Day 4–5:** Implement chosen improvement (prompt eng or LoRA)
4. **Day 6:** Full evaluation, all metrics, error taxonomy, ablation table
5. **Day 7:** Write report, polish figures, citations

## Confirmed Decisions
1. **GPU:** Available — full LoRA training feasible
2. **Team:** Solo
3. **Improvement method:** LoRA fine-tuning
4. **Model:** Qwen2.5-VL-3B-Instruct
5. **Deliverable:** PDF report + code repo

See `tech-stack.md` for the implementation details, repo layout, and shell-script entrypoints.
