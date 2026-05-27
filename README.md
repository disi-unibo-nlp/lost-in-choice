# Lost in Choice: Evaluating LLMs Under Extreme Multiple-Choice Scaling

> Anonymous ACL submission

Research project providing the first systematic investigation of LLM selection dynamics under extreme candidate cardinality. We stress-test six reasoning models by scaling the candidate space up to 1,000 options while manipulating their layout within the context window, and we audit Chain-of-Thought traces to characterize the behavioral heuristics models adopt as the candidate space expands.

## The Lost-in-Choice Phenomenon

Standard multiple-choice benchmarks (MMLU-Pro: 10 options) make it hard to disentangle two failure modes:

1. **Reasoning failure** — the model cannot derive the correct answer.
2. **Selection failure** — the model is overwhelmed by the size or position of the option list.

We find a systematic, layout-driven degradation of global comparative reasoning: models that perform near ceiling on 10 options can collapse to near-random at 1,000 — a failure mode we define as the *Lost-in-Choice* phenomenon.

## Benchmark: MMLU-1K

A diagnostic benchmark of **1,000 questions** from MMLU-Pro (CS, Health, Law, Math — 250 each), expanded to **31,000 unique test instances** by varying three independent axes:

| Axis | Values |
|---|---|
| **Cardinality** (N) | 100, 500, 1000 (+ 10-option baseline) |
| **Spatial layout** | Block (original 10 options contiguous) · Sparse (spread one-per-segment) |
| **Target position** (w) | 0.0, 0.25, 0.50, 0.75, 1.0 |

3 cardinalities × 2 layouts × 5 positions = **30 variants per question**, plus the 10-option baseline.

### Question typology

We formalize a key distinction to decouple knowledge retrieval from option-comparison complexity:

- **Option-Independent**: Answer derivable from internal knowledge prior to inspecting candidates (e.g., "What is the capital of Italy?"). 540 questions.
- **Option-Dependent**: Correct choice emerges only through comparative analysis of the available alternatives (e.g., "Which of the following is most efficient?"). 460 questions.

| Domain | Opt-Indep | Opt-Dep | Total |
|---|:---:|:---:|:---:|
| Computer Science | 204 | 46 | 250 |
| Health | 107 | 143 | 250 |
| Law | 57 | 193 | 250 |
| Mathematics | 172 | 78 | 250 |
| **Total** | **540** | **460** | **1,000** |

### Distractor quality control

Distractors are sampled in-domain (same MMLU-Pro category) to avoid semantic hardening. We screen all 990,000 unique question-distractor pairs via a dual-judge pipeline (GPT-OSS-120B for CS/Law/Health; Gemini-3-Flash for Math). 51.5% of questions contained ≥ 1 defensible distractor; 2,649 replacements were applied across the benchmark via a direct-replacement + safe-fallback protocol.

## Models

Six reasoning models (all emit CoT traces in `<think>…</think>`):

| Model | Access | Notes |
|---|---|---|
| Qwen3.5-27B | Local (vLLM) | FP8; `max_tokens=32000` |
| Qwen3.5-9B | Together AI | FP8; `max_tokens=32000` |
| Gemini-3-Flash | Vertex AI Batch API | `thinkingLevel=high` |
| Gemma-4-31B | Together AI | FP8; `max_tokens=32000` |
| GPT-OSS-120B | Together AI | `reasoning_effort=medium` |
| GPT-OSS-20B | Together AI | `reasoning_effort=medium` |

All responses use structured JSON output (`{"answer_index": <int>}`) to eliminate parsing noise.

## Metrics

**Accuracy (Acc)** — frequency of selecting the correct gold label out of N candidates.

**Reasoning Retention Rating (R3)** — aggregate resilience index synthesizing performance across all scaled cardinalities, layouts, and target positions relative to the 10-option baseline. R3 = 80% means the model retains 80% of its native capacity under severe structural manipulation. Computed via a two-level dynamic weighting scheme (within-window resilience × cross-window loss projection; see Appendix E).

**Error breakdown** — wrong predictions split into *Wrong Original* (one of the 9 native distractors) and *Random Distractor* (an injected filler), used to measure semantic anchoring vs. noise susceptibility.

## Key Findings

- Accuracy degrades universally as N scales; degradation is steeper on Option-Dependent tasks and in Sparse mode.
- Positional bias is pronounced: the gold answer placed at the very end of the context (w=1.0) suffers the steepest drop. Performance curves are not U-shaped — there is no recovery at the tail.
- Model hierarchy by R3: **Gemini-3-Flash > Qwen3.5-27B > Gemma-4-31B > Qwen3.5-9B > GPT-OSS-120B > GPT-OSS-20B**.
- Two architectural archetypes: *Heavy Reasoners* (Qwen3.5, Gemini-3-Flash; 6K–10K tokens) mitigate scaling drops; *Light Reasoners* (GPT-OSS, Gemma-4-31B; < 3K tokens) collapse under high context density.
- Strategy selection is driven by task dependency, not model scale: Option-Independent tasks are dominated by S1 (78–91%); Option-Dependent tasks redistribute mass to interactive strategies S2–S4 (8–35%).

### Reasoning strategy taxonomy

Derived from a CoT audit of ~stratified sample of traces, classified by an LLM-based classifier:

| Code | Name | Description |
|---|---|---|
| **S1** | Independent Derivation then Scan | Derives answer from knowledge, then confirms against the list |
| **S2** | Sequential Review | Linear option-by-option evaluation |
| **S3** | Eliminative Filtering | Discards implausible candidates progressively |
| **S4** | Candidate-Led Shortlisting | Generates internal hypotheses, matches against list |
| **S8** | Block/Cluster Detection | Identifies the contiguous block of original options and exploits local structure to narrow the search (prominent in Block mode) |
| **Other** | — | Rare or mixed behaviors |

---

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
pip install -r requirements.txt
```

For local inference with Qwen3.5-27B (requires CUDA):
```bash
pip install vllm==0.19.0
```

Create a `.env` file:
```
TOGETHER_API_KEY=...
GOOGLE_API_KEY=...
```

---

## Pipeline

The full pipeline runs in three phases. See [`preprocess/README.md`](preprocess/README.md) for detailed instructions, all script arguments, and intermediate output formats.

### Phase 1 — Build MMLU-1K (`preprocess/`)

| Step | Script | Output |
|------|--------|--------|
| 1 | `sample_mmlu.py` | `mmlu_pro_downsampled_2858/` |
| 2 | `categorize_mmlu.py` | batch job per judge model |
| 3 | `retrieve_categorize_mmlu.py` | `mmlu_pro_downsampled_2858_categorized/<model>/` |
| 4 | `combine_predictions.py` | `mmlu_pro_downsampled_2858_categorized_combined/` — majority vote labels |
| 5 | `sample_final_dataset.py` | `data/bench/mmlu-pro-augmented-10/` — the 1,000-question core |
| 6 | `augment_final_dataset_v4.py` | `data/bench/mmlu-pro-augmented-{100,500,1000}/` — 30 variants each |

Steps 2–3 are repeated for each of the three judge models (GPT-OSS-120B, Gemma-4-31B, Qwen3.5-9B). Final labels are assigned by majority vote.

### Phase 1b — Distractor quality control (`preprocess/`)

Screens all 990,000 question-distractor pairs for coincidental correctness and applies replacements before inference.

| Step | Script | Output |
|------|--------|--------|
| 7 | `check_distractors.py` / `check_distractor_gemini.py` | batch job per domain |
| 8 | `retrieve_distractor_check.py` / `retrieve_distractor_check_gemini.py` | `out/distractor_check/<model>/<domain>/<timestamp>/flagged_distractors.jsonl` |
| 9 | `merge_distractor_corrections.py` | updated `flagged_distractors.jsonl` (parse-failure fixes merged) |
| 10 | `merge_judge_outputs.py` | `out/distractor_check/merged/block/flagged_distractors.jsonl` |
| 11 | `apply_distractor_fixes.py` | `data/bench/mmlu-pro-augmented-clean-{100,500,1000}/` |

GPT-OSS-120B judges CS, Law, and Health; Gemini-3-Flash judges Mathematics. 51.5% of questions have ≥ 1 flagged distractor (2,649 replacements total). False-negative rate < 4.5% (human audit on 200 samples, Cohen's κ = 0.82).

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

Inference reads from `data/bench/mmlu-pro-augmented-clean-{N}/`. Each submitted job covers one (model, window, placement mode, seed) combination — 1,000 questions. Qwen3.5-27B is served locally via vLLM; Gemini-3-Flash via the Vertex AI Batch API; all others via the Together AI Batch API.

### Phase 3 — Analysis & Figures (`src/analysis/`)

After inference, use the scripts in `src/analysis/` to produce all paper figures and tables. See [`src/analysis/README.md`](src/analysis/README.md) for the full list with usage examples.

Typical order:
1. `core/analyze_reasoning.py` — aggregate token usage → `reasoning_records.jsonl`
2. `taxonomy/` pipeline — open-code CoT traces → synthesize taxonomy → classify → `classified_traces.jsonl`
3. `core/analyze_strategies.py` — build `strategy_stats.json`
4. `plots/` scripts — produce PDFs
5. `tables/` scripts — produce LaTeX

## Experimental Conditions

| Comparison | What it reveals |
|---|---|
| Accuracy vs. window position (same mode) | Primacy / recency bias |
| Block vs. Sparse at the same window | Effect of local option clustering |
| Option-Dependent vs. Option-Independent | Whether answer derivability interacts with positional bias |
| Per-domain breakdown across windows | Domain-specific sensitivity to option position |
| N = 100 vs. 500 vs. 1000 | How bias scales with option list length |
| R3 across models | Global resilience ranking |
