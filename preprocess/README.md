# Preprocessing Pipeline

This directory contains all scripts to build the benchmark dataset from scratch, starting from the raw MMLU-Pro HuggingFace dataset and ending with quality-controlled augmented variants ready for inference.

Run scripts in the order listed below. Steps 2–3 must be repeated for each judge model used in categorization.

---

## Pipeline Overview

```
TIGER-Lab/MMLU-Pro (HuggingFace)
        │
        ▼
sample_mmlu.py
        │  Filters to 4 categories × all questions with 10 options
        ▼
mmlu_pro_downsampled_2858/              ← 2858 questions (HF dataset)
        │
        ▼  (run for each judge model)
categorize_mmlu.py  ──────────────────► out/completions/<model>/categorization/<timestamp>/
retrieve_categorize_mmlu.py                      job_id.txt, raw_completions.jsonl
        │
        ▼
mmlu_pro_downsampled_2858_categorized/
  gpt-oss-120b/                         ← one subfolder per judge model (HF dataset)
  gemma-4-31B-it/
  ...
        │
        ▼
combine_predictions.py
        │  Majority vote across judges (gpt-oss-120b as primary tiebreaker)
        ▼
mmlu_pro_downsampled_2858_categorized_combined/  ← 2858 questions with final labels
        │
        ▼
sample_final_dataset.py
        │  All 460 option_dependent + 540 option_independent (balanced per category)
        ▼
data/bench/mmlu-pro-augmented-10/       ← 1000-question benchmark, 10 options each
        │
        ▼
augment_final_dataset.py
        │  5 windows × 2 modes × 3 seeds = 30 variants, N ∈ {100, 500, 1000}
        ▼
data/bench/mmlu-pro-augmented-{100,500,1000}/
  seed_42/window_0.00/block/
  seed_42/window_0.00/sparse/
  ...
        │
        ▼  ── DISTRACTOR QUALITY CONTROL ──────────────────────────────────────
        │
        ▼  (GPT-oss-120b for CS / law / health;  Gemini 2.0 Flash for math)
check_distractors.py / check_distractor_gemini.py
retrieve_distractor_check.py / retrieve_distractor_check_gemini.py
        │
        ▼
merge_distractor_corrections.py         ← fold in manually-corrected parse failures
merge_judge_outputs.py                  ← combine GPT and Gemini results by domain
        │
        ▼
out/distractor_check/merged/block/flagged_distractors.jsonl
        │
        ▼
apply_distractor_fixes.py
        │  Replace flagged distractors; fallback to safe pool on collision
        ▼
data/bench/mmlu-pro-augmented-clean-{100,500,1000}/
  seed_42/window_0.00/block/
  seed_42/window_0.00/sparse/
  ...                                   ← quality-controlled variants (used for inference)
```

---

## Step 1 — Download and Filter MMLU-Pro

**Script**: `sample_mmlu.py`

Downloads TIGER-Lab/MMLU-Pro from HuggingFace and retains all questions from the four target domains that have exactly 10 answer options.

```bash
python preprocess/sample_mmlu.py
```

**Configuration** (edit at the top of the script):

| Variable | Default | Description |
|---|---|---|
| `target_categories` | `['computer science', 'math', 'health', 'law']` | Domains to include |
| `sample_data_per_category` | `False` | If `True`, cap each category at `samples_per_category` |
| `samples_per_category` | `250` | Cap per category (only active if flag above is `True`) |
| `seed` | `42` | Shuffle seed |

**Output**: `mmlu_pro_downsampled_2858/` (HuggingFace dataset directory)

---

## Step 2 — Categorize Questions (per judge model)

**Script**: `categorize_mmlu.py`

Submits a Together AI batch job to classify each question as `option_dependent` or `option_independent`. Each call creates one batch job for one model.

```bash
python preprocess/categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --model-name "openai/gpt-oss-120b" \
  --output-dir out/completions \
  --limit 100    # omit for full run
```

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--dataset-dir` | `mmlu_pro_downsampled_2858` | Path to the HF dataset from step 1 |
| `--model-name` | `qwen/Qwen3.5-9B` | Together AI model path |
| `--output-dir` | `out/completions` | Base directory for outputs |
| `--limit` | `None` | Cap number of questions (for smoke tests) |

**Output**: `out/completions/<model>/categorization/<timestamp>/`
- `batch_together_<model>.jsonl` — uploaded batch requests
- `job_id.txt` — Together AI job ID (needed for step 3)

Repeat this step for each judge model (e.g. `gpt-oss-120b`, `gemma-4-31B-it`).

---

## Step 3 — Retrieve Categorization Results (per judge model)

**Script**: `retrieve_categorize_mmlu.py`

Polls the batch job, downloads completions, parses labels, and saves the augmented dataset. Run once per judge model after the job completes.

```bash
python preprocess/retrieve_categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --output-dir out/completions/gpt-oss-120b/categorization/<timestamp> \
  --job-id <BATCH_JOB_ID>
```

**Output**: `mmlu_pro_downsampled_2858_categorized/<model_name>/` (HF dataset + JSONL)

---

## Step 4 — Combine Judge Predictions

**Script**: `combine_predictions.py`

Merges predictions from all judge models via majority voting. `gpt-oss-120b` is the primary tiebreaker.

```bash
python preprocess/combine_predictions.py \
  --categorized-dir mmlu_pro_downsampled_2858_categorized \
  --base-dataset-dir mmlu_pro_downsampled_2858 \
  --output-path mmlu_pro_downsampled_2858_categorized_combined
```

**Output**: `mmlu_pro_downsampled_2858_categorized_combined/` (HF dataset + JSONL)

---

## Step 5 — Build the 1000-Question Benchmark

**Script**: `sample_final_dataset.py`

Builds the curated benchmark: all 460 `option_dependent` questions + 540 `option_independent` questions balanced to 250 per category.

```bash
python preprocess/sample_final_dataset.py \
  --combined-path mmlu_pro_downsampled_2858_categorized_combined \
  --output-path data/bench/mmlu-pro-augmented-10
```

**Output**: `data/bench/mmlu-pro-augmented-10/` (HF dataset + JSONL)

| Category         | option_dependent | option_independent | Total |
|------------------|:----------------:|:------------------:|:-----:|
| computer science |        46        |        204         |  250  |
| health           |       143        |        107         |  250  |
| law              |       193        |         57         |  250  |
| math             |        78        |        172         |  250  |
| **Total**        |     **460**      |      **540**       | **1000** |

---

## Step 6 — Augment to N Options

**Script**: `augment_final_dataset.py`

Expands each question from 10 to N options across 30 variants (5 windows × 2 modes × 3 seeds) for each pool size N ∈ {100, 500, 1000}. Smaller pools are strict prefixes of larger ones (distractors(100) ⊆ distractors(500) ⊆ distractors(1000)).

```bash
python preprocess/augment_final_dataset.py \
  --dataset-path data/bench/mmlu-pro-augmented-10 \
  --source-dataset-path mmlu_pro_downsampled_2858 \
  --n-options 100 500 1000 \
  --seeds 42 123 456
```

**Window positions** — gold answer lands at `block_start + answer_index`:

| Window | Block start (N=1000) | Gold index range |
|:------:|:--------------------:|:----------------:|
| 0.00   | 0                    | [0, 9]           |
| 0.25   | 247                  | [247, 256]       |
| 0.50   | 495                  | [495, 504]       |
| 0.75   | 742                  | [742, 751]       |
| 1.00   | 990                  | [990, 999]       |

**Placement modes**:
- `block` — original 10 options as a contiguous block at `block_start`.
- `sparse` — gold at the same absolute index as block; other 9 originals spread one-per-segment.

**Output**: `data/bench/mmlu-pro-augmented-{N}/seed_<s>/window_<w>/{block,sparse}/`

Each record adds: `new_options`, `new_answer_index`, `block_start_index`, `window`, `placement_mode`.

---

## Steps 7–8 — Distractor Quality Check

**Scripts**: `check_distractors.py` (Together AI / GPT) and `check_distractor_gemini.py` (Google Gemini)

Randomly sampled distractors may be defensible answers to their question, which would confound positional-bias measurements. These scripts submit LLM judge jobs that check each distractor for correctness and propose replacements.

Distractors are batched in groups of 100 per question. The judge returns a structured JSON response listing flagged items with reasons and replacement suggestions.

**Model assignment**:
- GPT-oss-120b → computer science, law, health
- Gemini 2.0 Flash Preview → mathematics (better handling of numerical distractors)

```bash
# GPT-oss-120b (all domains, used for CS / law / health)
bash scripts/check_distractors.sh

# Gemini (math domain only)
bash scripts/check_distractor_gemini.sh
```

After jobs complete, retrieve results:

```bash
bash scripts/retrieve_distractor_check.sh
bash scripts/retrieve_distractor_check_gemini.sh
```

**Output**: `out/distractor_check/<model>/<domain>/<timestamp>/`
- `flagged_distractors.jsonl` — one record per question with flagged items
- `summary.json` — aggregate statistics
- `raw_completions.jsonl` — raw API responses

---

## Step 9 — Merge Parse-Failure Corrections

**Script**: `merge_distractor_corrections.py`

Any batch responses that failed to parse are manually reviewed and stored as `fails_corrected.json` in the job output directory. This script folds those corrections back into the main `flagged_distractors.jsonl`.

```bash
python preprocess/merge_distractor_corrections.py \
  --output-dir out/distractor_check/<model>/<domain>/<timestamp> \
  --data data/bench/mmlu-pro-augmented-1000/seed_42/window_0.00/block.jsonl
```

---

## Step 10 — Merge Judge Outputs by Domain

**Script**: `merge_judge_outputs.py`

Combines the GPT and Gemini `flagged_distractors.jsonl` files into a single merged file, substituting the Gemini results for the math domain.

```bash
python preprocess/merge_judge_outputs.py \
  --base     out/distractor_check/openai_gpt-oss-120b/block/2026-04-24_15-04-12/flagged_distractors.jsonl \
  --override out/distractor_check/gemini-3-flash-preview/block/math/2026-04-25_01-44-10/flagged_distractors.jsonl \
  --category math \
  --output   out/distractor_check/merged/block/flagged_distractors.jsonl
```

**Flagging summary** (post-merge):

| Domain           | Flagged questions | Avg flags/q | Avg flags/flagged-q | Dup% |
|------------------|:-----------------:|:-----------:|:-------------------:|:----:|
| Computer Science | 107 / 250 (42.8%) |    0.96     |        2.24         | 1.7% |
| Mathematics      | 128 / 250 (51.2%) |    5.79     |       11.30         | 6.4% |
| Law              | 153 / 250 (61.2%) |    2.24     |        3.66         | 1.0% |
| Health           | 127 / 250 (50.8%) |    1.61     |        3.17         | 1.0% |
| **Overall**      | **515 / 1000 (51.5%)** | **2.65** | **5.14**       | **2.5%** |

*Dup%: fraction of replacement suggestions that are duplicates within the same question.*

---

## Step 11 — Apply Distractor Fixes

**Script**: `apply_distractor_fixes.py` (or `bash scripts/apply_distractor_fixes.sh`)

Applies the replacement map to all 30 variants for each pool size. Replacements are matched by distractor text, making them position-independent and valid across all window/mode combinations.

**Fallback logic**: when the judge's suggested replacement text matches one of the original 10 options (which would break sparse-mode structural invariants), a safe alternative is drawn uniformly at random from the distractors that passed the judge's sweep without being flagged.

```bash
python preprocess/apply_distractor_fixes.py \
  --flagged-jsonl out/distractor_check/merged/block/flagged_distractors.jsonl \
  --seed 42 \
  --n-options 100 500 1000 \
  --output-base      data/bench/mmlu-pro-augmented \
  --output-base-out  data/bench/mmlu-pro-augmented-clean
```

**Output**: `data/bench/mmlu-pro-augmented-clean-{100,500,1000}/seed_42/` — structurally identical to the original augmented datasets; all sanity checks A–K pass.

At N=1000: 2,649 replacements applied (2,235 direct + 414 fallbacks), zero gold-answer collisions.

---

## Analysis Scripts (`analysis/`)

| Script | Purpose |
|---|---|
| `sanity_check_v4.py` | Full structural verification of augmented datasets (checks A–K) |
| `distractor_intervention_stats.py` | Flag-rate statistics across domains |
| `pilot_distractor_judge.py` | Pilot evaluation of judge model candidates |
| `explore_dataset.py` | Category and label distributions |
| `check_reproducibility.py` | Verify deterministic augmentation across runs |
| `inspect_batch_alignment.py` | Check alignment between uploaded requests and retrieved completions |

## Utility Scripts (`preprocess/`)

| Script | Purpose |
|---|---|
| `distractor_correction_stats.py` | Print and export per-domain correction statistics |
| `export_corrections_table.py` | Generate LaTeX table of correction examples for paper appendix |
