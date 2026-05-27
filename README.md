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

Each question is augmented to N options (N ∈ {100, 500, 1000}) across **30 variants** (5 windows × 2 modes × 3 seeds). Distractors are drawn from same-category option texts in the source dataset. Before inference, all distractors are validated by an LLM judge and any defensible answers are replaced — see Phase 1b below.

## Repository Structure

```
preprocess/          ← dataset construction + distractor QC pipeline (see preprocess/README.md)
  analysis/          ← sanity checks, stats, exploration
src/                 ← experiment code (inference + evaluation)
  analysis/          ← post-processing: figures, tables, strategy taxonomy (see src/analysis/README.md)
    core/            ← data aggregation & statistics
    plots/           ← paper figures (PDF)
    tables/          ← LaTeX tables
    taxonomy/        ← reasoning-strategy taxonomy pipeline
  utils/             ← result summarization helpers
scripts/             ← shell helpers for batch jobs
notes/               ← research notes and design documents
data/                ← datasets (gitignored, populated by pipeline)
out/                 ← batch job outputs (gitignored)
```

## Setup

```bash
pip install datasets together python-dotenv google-genai
```

Create a `.env` file:
```
TOGETHER_API_KEY=...
GOOGLE_API_KEY=...
```

## Pipeline

The full pipeline runs in three phases. See [`preprocess/README.md`](preprocess/README.md) for detailed instructions, all script arguments, and intermediate output formats.

### Phase 1 — Build the benchmark (`preprocess/`)

| Step | Script | Output |
|------|--------|--------|
| 1 | `sample_mmlu.py` | `mmlu_pro_downsampled_2858/` |
| 2 | `categorize_mmlu.py` | batch job per judge model |
| 3 | `retrieve_categorize_mmlu.py` | `mmlu_pro_downsampled_2858_categorized/<model>/` |
| 4 | `combine_predictions.py` | `mmlu_pro_downsampled_2858_categorized_combined/` |
| 5 | `sample_final_dataset.py` | `data/bench/mmlu-pro-augmented-10/` |
| 6 | `augment_final_dataset_v4.py` | `data/bench/mmlu-pro-augmented-{100,500,1000}/` (30 variants each) |

Steps 2–3 are repeated for each judge model used in categorization.

### Phase 1b — Distractor quality control (`preprocess/`)

Synthetically generated distractors may occasionally be defensible answers. This phase identifies and replaces them before inference to avoid confounding positional-bias measurements.

| Step | Script | Output |
|------|--------|--------|
| 7 | `check_distractors.py` / `check_distractor_gemini.py` | batch job per domain |
| 8 | `retrieve_distractor_check.py` / `retrieve_distractor_check_gemini.py` | `out/distractor_check/<model>/<domain>/<timestamp>/flagged_distractors.jsonl` |
| 9 | `merge_distractor_corrections.py` | updated `flagged_distractors.jsonl` (parse-failure fixes merged) |
| 10 | `merge_judge_outputs.py` | `out/distractor_check/merged/block/flagged_distractors.jsonl` |
| 11 | `apply_distractor_fixes.py` | `data/bench/mmlu-pro-augmented-clean-{100,500,1000}/` |

GPT-oss-120b is used as judge for computer science, law, and health; Gemini 2.0 Flash Preview for mathematics. Overall 51.5% of questions have ≥ 1 flagged distractor (2,649 replacements applied across the benchmark).

### Phase 2 — Run inference (`src/`)

```bash
# Submit a batch inference job for one configuration
python src/run_inference.py \
  --window 0.5 --mode sparse \
  --model-name "openai/gpt-oss-120b" \
  --seed 42

# Retrieve results and compute accuracy
python src/retrieve_inference.py \
  --output-dir out/completions/gpt-oss-120b/<dataset_tag>/<timestamp>
```

Inference reads from `data/bench/mmlu-pro-augmented-clean-{N}/` (the quality-controlled dataset).

## Phase 3 — Analysis & Figures (`src/analysis/`)

After inference, use the scripts in `src/analysis/` to produce all paper figures and tables.
See [`src/analysis/README.md`](src/analysis/README.md) for the full list with usage examples.

Typical order:
1. `core/analyze_reasoning.py` — aggregate token usage → `reasoning_records.jsonl`
2. `taxonomy/` pipeline — open-code traces → classify → `classified_traces.jsonl`
3. `core/analyze_strategies.py` — build `strategy_stats.json`
4. `plots/` scripts — produce PDFs
5. `tables/` scripts — produce LaTeX

## Experimental Conditions

| Comparison | What it reveals |
|---|---|
| Accuracy vs. window position (same mode) | Primacy / recency bias |
| Block vs. sparse at the same window | Effect of local option clustering |
| option_dependent vs. option_independent | Whether answer derivability interacts with positional bias |
| Per-category breakdown across windows | Domain-specific sensitivity to option position |
| N = 100 vs. 500 vs. 1000 | How bias scales with option list length |
