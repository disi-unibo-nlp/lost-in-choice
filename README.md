# Lost in Choice: LLMs Fail to Reason Under Large-Scale Multiple-Choice Evaluation

Research project studying how LLMs behave when the number of answer options scales from 10 to 1000, with systematic control over where the correct answer appears and how the original options are arranged.

## Motivation

Standard multiple-choice benchmarks (MMLU-Pro: 10 options) make it hard to disentangle two failure modes:

1. **Reasoning failure** — the model cannot derive the correct answer.
2. **Selection failure** — the model is distracted by the size or position of the option list.

We expand each question to **1000 options** and vary two independent axes:
- **Gold answer position** — 5 sliding window positions (0%, 25%, 50%, 75%, 100% through the list).
- **Structural arrangement** — `block` (original 10 options contiguous) vs `sparse` (spread one-per-segment).

A well-calibrated model should show flat accuracy across all window positions. Any curve is evidence of positional heuristics rather than reasoning.

## Dataset Design

Starting from MMLU-Pro (4 domains: computer science, health, law, math), we build a curated **1000-question benchmark** with two constraints:

1. All 460 `option_dependent` questions are included (questions where the correct answer cannot be derived without reading the options).
2. The remaining 540 slots are filled with `option_independent` questions, balanced to 250 per category.

| Category         | option_dependent | option_independent | Total |
|------------------|:----------------:|:------------------:|:-----:|
| computer science |        46        |        204         |  250  |
| health           |       143        |        107         |  250  |
| law              |       193        |         57         |  250  |
| math             |        78        |        172         |  250  |

Each question is then augmented to 1000 options across **30 variants** (5 windows × 2 modes × 3 seeds). Distractors are drawn from same-category option texts in the source dataset, excluding the current question's own 10 options.

## Repository Structure

```
preprocess/                    ← dataset construction pipeline
  categorize_mmlu.py           ← submit categorization batch job (Together AI)
  retrieve_categorize_mmlu.py  ← retrieve categorization results
  combine_predictions.py       ← majority-vote labels across judges
  sample_final_dataset.py      ← build the 1000-question benchmark
  augment_final_dataset_v2.py  ← expand to N options (windows × modes × seeds)
  analysis/                    ← sanity checks, stats, exploration
src/                           ← experiment code
  run_inference.py             ← submit inference batch job (Together AI)
  retrieve_inference.py        ← retrieve results + compute accuracy metrics
  utils/
    summarize_results.py
scripts/                       ← shell helpers
notes/                         ← research notes and design documents
data/                          ← datasets (gitignored)
out/                           ← batch job outputs (gitignored)
```

## Setup

```bash
pip install datasets together python-dotenv
```

Create a `.env` file:
```
TOGETHER_API_KEY=...
```

## Running the Pipeline

```bash
# 1. Categorize questions (option_dependent vs option_independent)
python preprocess/categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --model-name "openai/gpt-oss-120b" \
  --output-dir out/completions

# 2. Retrieve categorization results
python preprocess/retrieve_categorize_mmlu.py \
  --dataset-dir mmlu_pro_downsampled_2858 \
  --output-dir out/completions/gpt-oss-120b/categorization/<timestamp> \
  --job-id <JOB_ID>

# 3. Combine multi-judge predictions
python preprocess/combine_predictions.py

# 4. Build the 1000-question benchmark
python preprocess/sample_final_dataset.py \
  --combined-path mmlu_pro_downsampled_2858_categorized_combined \
  --output-path data/bench/mmlu-pro-augmented-10

# 5. Augment to 1000 options (30 variants)
python preprocess/augment_final_dataset_v2.py \
  --dataset-path data/bench/mmlu-pro-augmented-10 \
  --source-dataset-path mmlu_pro_downsampled_2858

# 6. Run inference
python src/run_inference.py \
  --window 0.5 --mode sparse \
  --model-name "openai/gpt-oss-120b" \
  --seed 42

# 7. Retrieve results and compute accuracy
python src/retrieve_inference.py \
  --output-dir out/completions/gpt-oss-120b/<dataset_tag>/<timestamp>
```

## Experimental Conditions

| Comparison | What it reveals |
|---|---|
| Accuracy vs. window position (same mode) | Primacy / recency bias |
| Block vs. sparse at the same window | Effect of local option clustering |
| option_dependent vs. option_independent | Whether answer derivability interacts with positional bias |
| Per-category breakdown across windows | Domain-specific sensitivity to option position |
