# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying positional bias in LLMs under large-scale multiple-choice evaluation. The pipeline categorizes MMLU-Pro questions by option dependency, builds a curated 1000-question benchmark, augments each question to 1000 options across 5 sliding window positions × 2 placement modes, and evaluates models via the Together AI batch API.

## Repository Structure

```
preprocess/          ← dataset construction pipeline
  analysis/          ← sanity checks, stats, exploration
src/                 ← experiment code (inference + evaluation)
  utils/             ← result summarization helpers
scripts/             ← shell helpers for batch jobs
data/                ← datasets (gitignored, populated by pipeline)
logs/                ← script output logs (gitignored)
notes/               ← research notes and design documents
```

## Pipeline Stages

Run these scripts in order:

```bash
# 1. Submit batch categorization job (option_dependent vs option_independent)
python preprocess/categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --model-name "openai/gpt-oss-120b" \
  --output-dir out/completions \
  --limit 100   # omit for full run

# 2. Retrieve categorization results (once job completes)
python preprocess/retrieve_categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --output-dir out/completions/gpt-oss-120b/categorization/<timestamp> \
  --job-id <BATCH_JOB_ID>

# 3. Combine multi-judge predictions via majority vote
python preprocess/combine_predictions.py

# 4. Build final 1000-question benchmark (balanced dep/indep × category)
python preprocess/sample_final_dataset.py \
  --combined-path mmlu_pro_downsampled_2858_categorized_combined \
  --output-path data/bench/mmlu-pro-augmented-10

# 5. Augment to 1000 options (5 windows × 2 modes × 3 seeds = 30 variants)
python preprocess/augment_final_dataset_v2.py \
  --dataset-path data/bench/mmlu-pro-augmented-10 \
  --source-dataset-path mmlu_pro_downsampled_2858 \
  --n-options 1000

# 6. Submit inference batch job
python src/run_inference.py \
  --window 0.5 \
  --mode sparse \
  --model-name "openai/gpt-oss-120b" \
  --seed 42 \
  --limit 10   # omit for full run

# 7. Retrieve results and compute accuracy
python src/retrieve_inference.py \
  --output-dir out/completions/gpt-oss-120b/<dataset_tag>/<timestamp>
  # --job-id is optional if job_id.txt is present in output-dir
```

## Architecture

**Data flow**: TIGER-Lab/MMLU-Pro (HuggingFace) → 2858-question downsample → Together AI batch API for categorization → majority-voted labels → curated 1000-question benchmark → augmented 1000-option variants → Together AI batch API for model evaluation → accuracy metrics.

**Batch jobs**: `categorize_mmlu.py` and `src/run_inference.py` submit jobs asynchronously to Together AI's batch API. Jobs return a job ID; results are retrieved later. Raw completions are stored under `out/completions/<model>/<dataset_tag>/<timestamp>/`.

**Option augmentation** (`augment_final_dataset_v2.py`): Expands 10-option questions to N options (default 1000) across 5 sliding window positions (0.0, 0.25, 0.5, 0.75, 1.0) and 2 placement modes:
- `block` — original 10 options stay contiguous; block shifted so gold lands at target index.
- `sparse` — original 10 options spread one-per-segment; gold lands at same absolute index as block mode.

**Structured decoding** (`src/run_inference.py`): Uses `response_format = json_schema` to force `{"answer_index": <int>}` from the model, eliminating parsing noise.

**Evaluation** (`src/retrieve_inference.py`): Computes overall accuracy, per-category, per-dependency-label, per-segment predicted-index distribution, and option-type breakdown (gold / wrong original / distractor).

## Environment

Requires a `.env` file with:
```
TOGETHER_API_KEY=...
```

## Key Data Schema

MMLU-Pro fields used throughout: `question_id`, `question`, `options` (list), `answer` (letter A–J), `answer_index` (0-based int), `category`.

Categorization adds: `option_dependent_vs_independent` (`"option_dependent"` | `"option_independent"`).

Augmentation adds: `new_options` (list of N strings), `new_answer_index` (int), `block_start_index` (int), `window` (float), `placement_mode` (`"block"` | `"sparse"`).
