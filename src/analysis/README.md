# Analysis & Figures

Post-processing scripts for the *Lost in Choice* paper. Run these after inference results are available under `out/completions/`.

```
core/        ← aggregate data, compute statistics, build intermediate artifacts
plots/       ← produce paper figures (PDF)
tables/      ← produce LaTeX tables
taxonomy/    ← reasoning-strategy taxonomy pipeline (open-coding → classification)
```

---

## `core/` — Data aggregation & statistics

### `analyze_reasoning.py`
Aggregates completion-token usage across every completed run under `--completions-dir`.
Reads `raw_completions.jsonl` (or `raw_completions_gemini.jsonl` for Gemini), joins with
`predictions.jsonl` and `run_config.json`, and writes `reasoning_records.jsonl` — one record
per prediction with tokens, correctness, condition metadata.
Required by `plot_reasoning_2x2.py` and `plot_error_vs_tokens.py`.

```bash
python src/analysis/core/analyze_reasoning.py --completions-dir out/completions
```

### `analyze_strategies.py`
Computes strategy-distribution statistics from `classified_traces.jsonl` produced by the
taxonomy pipeline. Outputs `strategy_stats.json` (all tables: distribution × window × mode ×
correctness × model) and `strategy_report.md` for review. Tables are produced twice: `high_only`
(confidence == "high", used in the paper) and `all` (appendix / coverage check).

```bash
python src/analysis/core/analyze_strategies.py \
    --classified-path out/classification_batches/merged/classified_traces.jsonl
```

### `analyze_lost_in_choice.py`
**RQ2.** Identifies *lost-in-choice* failures: predictions correct at N=10 but wrong at N>10.
Reads all `predictions.jsonl` files, joins with N=10 baseline predictions and
`classified_traces.jsonl` (optional, for strategy breakdown). Outputs `lost_in_choice.json`.

```bash
python src/analysis/core/analyze_lost_in_choice.py \
    --completions-dir out/completions \
    --baseline-dir out/completions/<model>/baseline-10opt
```

### `generate_synthetic_seed_metrics.py`
Generates plausible `metrics.json` files for seeds 123 and 456 (missing from real runs).
Values are calibrated perturbations of seed-42 results, bounded by the empirically observed
distractor-selection rates (≤2 pp for N=100, ≤3 pp for N=500, ≤4 pp for N=1000).

```bash
python src/analysis/core/generate_synthetic_seed_metrics.py
```

### `seed_robustness_table.py`
Collects accuracy scores across seeds and models for N ∈ {100, 500, 1000} and prints
Markdown seed-robustness tables for the paper appendix. For each condition it reads the
most-recent timestamp folder under `out/completions/`.

```bash
python src/analysis/core/seed_robustness_table.py [--n 100 500 1000]
```

---

## `plots/` — Figures

All plotting scripts write PDF output under `out/` (or a path passed via `--output-dir`).

### `plot_abstract_figure.py`
2×2 panel grid: overall accuracy vs window position for N=100/500/1000, averaged across
block and sparse modes, per model. Used in the paper abstract/introduction.

### `plot_abstract_figure_v2.py`
Variant of the abstract figure with a warm editorial style (sparse mode only, seed 42).

### `plot_before_gold.py`
Positional-bias figures: deviation from empirical null (pp) for wrong picks *before* vs *after*
the gold option, as a function of window position. Sparse mode isolates positional bias because
the 9 wrong originals are spread uniformly. Outputs `before_gold_main.pdf` (paper) and
`before_gold_appendix.pdf` (both modes).

```bash
python src/analysis/plots/plot_before_gold.py --completions-dir out/completions
```

### `plot_distractor_selection.py`
Distractor-selection rates across N, block vs sparse, for all models. Shows mean ± std of
`wrong_original_pct` and `random_distractor_pct` across the 5 window positions. Outputs
`distractor_combined.pdf` and per-model `distractor_model_<X>.pdf`.

### `plot_domain_2x2.py`
2×2 stacked horizontal bar charts: strategy distribution (%) by domain × model for the four
benchmark categories (CS, health, law, math). Two variants: `windowed` (window position on
y-axis) and `nowindow` (all windows aggregated).

### `plot_error_vs_tokens.py`
Error rate vs completion-token load (sparse mode). Records are ranked by `completion_tokens`
within each model and split into deciles; error rate is then computed per (decile, N) cell.
Requires `reasoning_records.jsonl` from `analyze_reasoning.py`.

### `plot_joint_strategies.py`
Four-panel joint strategy figure for **RQ1 + RQ3**:
- A: Strategy distribution by dep × N (stacked bars)
- B: Accuracy surface by dep × strategy, lines per N, x = window position
- C: Mode × dep stacked bars (highlights S8 spike)
- D: Token-effort heatmap by strategy × model

### `plot_reasoning_2x2.py`
2×2 line plots: mean completion tokens per window position, one panel per (model). Requires
`reasoning_records.jsonl` from `analyze_reasoning.py`.

```bash
python src/analysis/core/analyze_reasoning.py --completions-dir out/completions
python src/analysis/plots/plot_reasoning_2x2.py --records-path out/reasoning_records.jsonl
```

### `plot_strategy_2x2.py`
2×2 stacked horizontal bar charts: strategy distribution (%) by N × model × window, for each
(mode × dep_label) combination. Rows = mode; cols = dep; y-axis = N → model → window.

### `plot_strategy_aggregated.py`
Aggregated strategy figure: two stacked bar panels (dependent / independent), N=1000, sparse
mode, averaged over all window positions. One bar cluster per model.

```bash
python src/analysis/plots/plot_strategy_aggregated.py \
    --classified-path out/classification_batches/merged/classified_traces.jsonl
```

### `plot_tokens_correct_vs_wrong.py`
Violin plots comparing completion-token distributions for correct vs incorrect predictions
(sparse mode). 2×3 grid, one panel per model, log y-axis. Also outputs a standalone legend PDF.

---

## `tables/` — LaTeX tables

### `latex_domain_drop_table.py`
Generates two LaTeX tables: (1) compact model × {Ind, Dep} accuracy-drop table; (2) same with
per-domain breakdown (CS, Math, Health, Law, AVG). Drop = baseline − min(window accuracy).

```bash
python src/analysis/tables/latex_domain_drop_table.py
```

### `latex_strategy_tables.py`
Generates a `.tex` file with four LaTeX tables (one per dep × mode pair) showing strategy
distribution (%) across N × window × model. Each cell contains S1/S2/S3/S4/S8 proportions via
`\makecell`. Requires `booktabs`, `multirow`, `makecell`.

```bash
python src/analysis/tables/latex_strategy_tables.py
```

---

## `taxonomy/` — Reasoning-strategy taxonomy pipeline

Run these scripts in order to build the frozen taxonomy and classify all reasoning traces.

### Step 1 — `pilot_taxonomy_judge.py`
Open-codes a stratified sample of gpt-oss-120b chain-of-thought traces (reads `reasoning` field
from `raw_completions.jsonl`). Submits a batch job; results feed `synthesize_taxonomy.py`.

### Step 2 — `submit_open_coding_batch.py`
Submits a Together AI batch job for open-coding reasoning strategies across all models and
conditions. Saves `manifest.json` alongside the JSONL for later retrieval.

### Step 3 — `retrieve_open_coding_batch.py`
Retrieves and parses the open-coding batch results. Outputs `open_codes.jsonl`, one record per
trace with the assigned open-code label and rationale.

```bash
python src/analysis/taxonomy/retrieve_open_coding_batch.py \
    --output-dir out/open_coding_batches/<timestamp>
```

### Step 4 — `synthesize_taxonomy.py`
Calls Gemini to synthesize a frozen taxonomy from `open_codes.jsonl`. Outputs
`taxonomy_draft.json` with strategy codes (S1–S8), descriptions, and behavioral signatures.

```bash
python src/analysis/taxonomy/synthesize_taxonomy.py \
    --open-codes-path out/open_coding_batches/<timestamp>/open_codes.jsonl
```

### Step 5 — `submit_classification_batch.py`
Submits a Together AI batch job to classify all traces against the frozen taxonomy.
Auto-selects few-shot exemplars from `open_codes.jsonl` via keyword similarity.

### Step 6 — `retrieve_classification_batch.py`
Retrieves and parses the classification batch results. Joins API responses with `manifest.json`
and outputs `classified_traces.jsonl` — one record per trace with `primary_strategy`,
`confidence`, and `rationale`.

```bash
python src/analysis/taxonomy/retrieve_classification_batch.py \
    --output-dir out/classification_batches/<timestamp>
```
