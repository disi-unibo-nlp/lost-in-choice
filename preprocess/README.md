# Preprocessing Pipeline

This directory contains all scripts to build the benchmark dataset from scratch, starting from the raw MMLU-Pro HuggingFace dataset and ending with the 1000-option augmented variants ready for inference.

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
augment_final_dataset_v2.py
        │  5 windows × 2 modes × 3 seeds = 30 variants
        ▼
data/bench/mmlu-pro-augmented-1000/
  seed_42/window_0.00/block/
  seed_42/window_0.00/sparse/
  ...
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

**Annotation prompt**: uses a structured dependency prompt with 5 worked examples covering edge cases (`which of the following`, `EXCEPT/NOT/LEAST`, composite True/False/Not-Given options). Uses regex-constrained decoding to force `option_dependent` or `option_independent`.

Repeat this step for each judge model (e.g. `gpt-oss-120b`, `gemma-4-31B-it`).

---

## Step 3 — Retrieve Categorization Results (per judge model)

**Script**: `retrieve_categorize_mmlu.py`

Polls the batch job, downloads completions, parses labels, and saves the augmented dataset. Run once per judge model after the job completes.

```bash
python preprocess/retrieve_categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --output-dir out/completions/gpt-oss-120b/categorization/2026-04-07_16-44-55 \
  --job-id <BATCH_JOB_ID>
```

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--dataset-dir` | `mmlu_pro_downsampled_2858` | Original dataset (for metadata) |
| `--output-dir` | — | Timestamp directory from step 2 |
| `--job-id` | — | Together AI job ID from `job_id.txt` |

**Output**: `mmlu_pro_downsampled_2858_categorized/<model_name>/` (HF dataset + JSONL)

Prints dependency statistics and a per-category breakdown after saving.

---

## Step 4 — Combine Judge Predictions

**Script**: `combine_predictions.py`

Merges predictions from all judge models in `mmlu_pro_downsampled_2858_categorized/` via majority voting. `gpt-oss-120b` is the primary tiebreaker.

**Voting rules**:
- 3 models: majority wins (≥ 2/3); all disagree → `gpt-oss-120b`
- 2 models (one missing): agree → use that; disagree → `gpt-oss-120b`
- 1 model: use `gpt-oss-120b` if available, else the only prediction

```bash
python preprocess/combine_predictions.py \
  --categorized-dir mmlu_pro_downsampled_2858_categorized \
  --base-dataset-dir mmlu_pro_downsampled_2858 \
  --output-path mmlu_pro_downsampled_2858_categorized_combined
```

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--categorized-dir` | `mmlu_pro_downsampled_2858_categorized` | Dir with one subfolder per judge model |
| `--base-dataset-dir` | `mmlu_pro_downsampled_2858` | Original dataset (for metadata) |
| `--output-path` | `mmlu_pro_downsampled_2858_categorized_combined` | Output path |

**Output**: `mmlu_pro_downsampled_2858_categorized_combined/` (HF dataset + JSONL)

Each record adds:
- `option_dependent_vs_independent` — final majority-voted label
- `vote_method` — `majority` / `primary_tiebreak` / `arbitrary_tiebreak`
- `pred_<model>` — per-model raw prediction

---

## Step 5 — Build the 1000-Question Benchmark

**Script**: `sample_final_dataset.py`

Builds the curated benchmark using two constraints:
1. All 460 `option_dependent` questions are included.
2. The remaining 540 slots are filled with `option_independent` questions, allocated to reach exactly 250 per category (via Hamilton's largest-remainder method).

```bash
python preprocess/sample_final_dataset.py \
  --combined-path mmlu_pro_downsampled_2858_categorized_combined \
  --output-path data/bench/mmlu-pro-augmented-10 \
  --total 1000 \
  --seed 42
```

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--combined-path` | `mmlu_pro_downsampled_2858_categorized_combined` | Output of step 4 |
| `--output-path` | `data/bench/mmlu-pro-augmented-10` | Output path |
| `--total` | `1000` | Target number of questions |
| `--seed` | `42` | Sampling seed |

**Output**: `data/bench/mmlu-pro-augmented-10/` (HF dataset + JSONL)

Final composition:

| Category         | option_dependent | option_independent | Total |
|------------------|:----------------:|:------------------:|:-----:|
| computer science |        46        |        204         |  250  |
| health           |       143        |        107         |  250  |
| law              |       193        |         57         |  250  |
| math             |        78        |        172         |  250  |
| **Total**        |     **460**      |      **540**       | **1000** |

---

## Step 6 — Augment to 1000 Options

**Script**: `augment_final_dataset_v2.py`

Expands each question from 10 to 1000 options across 30 variants (5 windows × 2 modes × 3 seeds). Distractors are drawn from same-category option texts in the 2858-question source pool, excluding the current question's own 10 options.

```bash
python preprocess/augment_final_dataset_v2.py \
  --dataset-path data/bench/mmlu-pro-augmented-10 \
  --source-dataset-path mmlu_pro_downsampled_2858 \
  --n-options 1000 \
  --seeds 42 123 456
```

**Arguments**:

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | `data/bench/mmlu-pro-augmented-10` | 1000-question benchmark from step 5 |
| `--source-dataset-path` | `mmlu_pro_downsampled_2858` | Full source dataset for distractor pools |
| `--output-root` | `data/bench/mmlu-pro-augmented-<n>` | Root output directory |
| `--n-options` | `1000` | Total options per question (must be divisible by 10) |
| `--seeds` | `42 123 456` | Distractor sampling seeds |

**Window positions** — the block of original options starts at `round(w × (N - 10))`:

| Window | Block start (N=1000) | Gold index range |
|:------:|:--------------------:|:----------------:|
| 0.00   | 0                    | [0, 9]           |
| 0.25   | 247                  | [247, 256]       |
| 0.50   | 495                  | [495, 504]       |
| 0.75   | 742                  | [742, 751]       |
| 1.00   | 990                  | [990, 999]       |

**Placement modes**:
- `block` — original 10 options kept as a contiguous block at `block_start`.
- `sparse` — gold answer at the same absolute index as block mode; the other 9 originals distributed one-per-segment across the remaining 9 segments of 100 slots each.

Block vs. sparse share identical gold positions — any accuracy difference is attributable purely to option clustering, not gold position.

**Output**: `data/bench/mmlu-pro-augmented-1000/seed_<N>/window_<w>/{block,sparse}/` (HF dataset + JSONL per variant)

Each augmented record adds:

| Field | Description |
|---|---|
| `new_options` | List of 1000 option strings |
| `new_answer_index` | Integer index of the correct answer |
| `block_start_index` | Index where the original 10-option block starts |
| `window` | Window fraction (0.0 / 0.25 / 0.5 / 0.75 / 1.0) |
| `placement_mode` | `"block"` or `"sparse"` |

---

## Analysis Scripts (`analysis/`)

These scripts are for inspection and sanity-checking intermediate outputs. They do not modify data.

| Script | Purpose |
|---|---|
| `explore_dataset.py` | Explore category and label distributions |
| `sanity_check_augmented.py` | Verify augmented option counts and gold positions |
| `sanity_check_v2.py` | Extended checks on v2 augmentation invariants |
| `check_reproducibility.py` | Verify that rerunning augmentation with the same seed gives identical results |
| `inspect_batch_alignment.py` | Check alignment between uploaded batch requests and retrieved completions |
