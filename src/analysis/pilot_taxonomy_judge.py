"""
Stage-1 taxonomy pilot: open-code reasoning strategies from gpt-oss-120b
chain-of-thought traces, then synthesize a draft taxonomy.

gpt-oss-120b is a reasoning model: every raw completion includes a
`message.reasoning` field with the full scratchpad. This script reads those
traces directly from raw_completions.jsonl and passes them to the judge as
primary evidence for strategy classification.

Sampling: n_per_condition traces per (n_options x window x mode), stratified
by correctness (half correct / half incorrect where possible).

Outputs (under --output-dir/<timestamp>/):
    open_codes.jsonl    — one record per trace with judge's inferred strategy
    taxonomy_draft.json — synthesized taxonomy (JSON)
    taxonomy_draft.md   — human-readable taxonomy table

Usage:
    python preprocess/analysis/pilot_taxonomy_judge.py
    python preprocess/analysis/pilot_taxonomy_judge.py \\
        --n-per-condition 10 \\
        --judge-model meta-llama/Llama-3.3-70B-Instruct-Turbo \\
        --output-dir out/taxonomy_pilot \\
        --dry-run
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import load_from_disk
from dotenv import load_dotenv
from pydantic import BaseModel
from together import Together

load_dotenv()

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
WINDOWS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODES = ["block", "sparse"]
INFERRED_SEED = 42
COMPLETIONS_ROOT = Path("out/completions")

# Default pool: model slug → N-options tiers it participates in
MODEL_POOL: dict[str, list[int]] = {
    #"gpt-oss-20b":  [ 100, 500, 1000],
    #"gemma-4-31B-it": [ 100, 500, 1000],
    "gpt-oss-120b": [ 100, 500, 1000],
    "Qwen3.5-9B":   [ 100, 500, 1000],
}


# -------------------------------------------------------------------------
# Structured output schemas
# -------------------------------------------------------------------------

class OpenCode(BaseModel):
    strategy_label: str        # 2–5 word label, e.g. "early sequential scan"
    strategy_description: str  # 1–2 sentence evidence-based explanation
    confidence: str            # "high" | "medium" | "low"


class TaxonomyEntry(BaseModel):
    code: str                   # e.g. "S1"
    name: str                   # short memorable name
    description: str            # 2–3 sentence definition
    behavioral_signature: str   # observable trace evidence that triggers this code
    frequency: str              # "high" (>30%) | "medium" (10–30%) | "rare" (<10%)


class TaxonomySynthesis(BaseModel):
    taxonomy: List[TaxonomyEntry]
    notes: str  # ambiguities, collisions, gaps


class Classification(BaseModel):
    primary_strategy: str           # one taxonomy code, e.g. "S1"
    secondary_strategy: Optional[str] = None  # follow-up code if two strategies clearly present
    confidence: str                 # "high" | "medium" | "low"
    rationale: str                  # 1-2 sentences citing specific trace phrases


# -------------------------------------------------------------------------
# Directory helpers
# -------------------------------------------------------------------------

def latest_run_dir(base: Path) -> Optional[Path]:
    """Most recent timestamped subdirectory that contains predictions.jsonl."""
    if not base.exists():
        return None
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and (d / "predictions.jsonl").exists()],
        reverse=True,
    )
    return dirs[0] if dirs else None


def condition_run_dir(model_slug: str, n_options: int, window: float, mode: str) -> Optional[Path]:
    tag = (
        f"mmlu-pro-augmented-{n_options}"
        f"__seed_{INFERRED_SEED}"
        f"__window_{window:.2f}"
        f"__{mode}"
    )
    return latest_run_dir(COMPLETIONS_ROOT / model_slug / tag)


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------

def load_predictions(run_dir: Path) -> dict:
    preds = {}
    with open(run_dir / "predictions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            preds[str(row["question_id"])] = row
    return preds


def load_dataset_by_qid(run_dir: Path) -> dict:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        cfg = json.load(f)
    dataset_path = cfg.get("dataset_path", "")
    if not Path(dataset_path).exists():
        return {}
    ds = load_from_disk(dataset_path)
    return {str(row["question_id"]): row for row in ds}


def load_reasoning_traces(run_dir: Path) -> dict:
    """Return {question_id: reasoning_text} from raw_completions.jsonl.

    gpt-oss-120b stores its scratchpad in message.reasoning.  Returns an
    empty string for any record where the field is absent or None.
    """
    path = run_dir / "raw_completions.jsonl"
    if not path.exists():
        return {}
    traces = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = str(row["custom_id"])
            try:
                reasoning = (
                    row["response"]["body"]["choices"][0]["message"].get("reasoning") or ""
                )
            except (KeyError, IndexError):
                reasoning = ""
            traces[qid] = reasoning
    return traces


# -------------------------------------------------------------------------
# Sampling
# -------------------------------------------------------------------------

def stratified_sample(predictions: dict, n: int, seed: int) -> list:
    """n/2 correct + n/2 incorrect; fills from the larger side if imbalanced."""
    correct = [v for v in predictions.values() if v["correct"] and not v.get("parse_failed")]
    incorrect = [v for v in predictions.values() if not v["correct"] and not v.get("parse_failed")]
    rng = random.Random(seed)
    n_c = min(n // 2, len(correct))
    n_i = min(n - n_c, len(incorrect))
    # fill slack from whichever side has more
    slack = n - n_c - n_i
    if slack > 0:
        if len(correct) > n_c:
            n_c = min(n_c + slack, len(correct))
        elif len(incorrect) > n_i:
            n_i = min(n_i + slack, len(incorrect))
    sampled = rng.sample(correct, n_c) + rng.sample(incorrect, n_i)
    rng.shuffle(sampled)
    return sampled


# -------------------------------------------------------------------------
# Trace context builder
# -------------------------------------------------------------------------

def build_context(pred: dict, ds_row: dict, reasoning: str = "", model_slug: str = "") -> dict:
    options = ds_row["new_options"]
    n = len(options)
    gold_idx = pred["gold_index"]
    pred_idx = pred["predicted_index"]
    return {
        "question_id": str(pred["question_id"]),
        "model_slug": model_slug,
        "question": ds_row["question"],
        "category": ds_row["category"],
        "option_dep": ds_row["option_dependent_vs_independent"],
        "n_options": n,
        "window": pred["window"],
        "placement_mode": pred["placement_mode"],
        "gold_index": gold_idx,
        "gold_option": options[gold_idx],
        "predicted_index": pred_idx,
        "predicted_option": options[pred_idx],
        "correct": pred["correct"],
        "hit_original_option": pred.get("hit_original_option"),
        "reasoning_trace": reasoning,
    }


# -------------------------------------------------------------------------
# Prompt construction
# -------------------------------------------------------------------------

OPEN_CODING_TEMPLATE = """\
You are analyzing how a large language model answered a multiple-choice question with {n_options} options.
The model is a reasoning model: its full chain-of-thought scratchpad is shown below.

Your task: identify the dominant reasoning strategy the model used, grounded in specific
evidence from the chain-of-thought.

---
QUESTION
{question}

Category: {category}
Option classification: {option_dep}

---
STRUCTURAL CONTEXT
  Total options: {n_options}
  Window position: {window}  (0.0 = gold near start, 0.5 = middle, 1.0 = near end)
  Placement mode: {mode} — {mode_desc}
  Hit an original (non-distractor) option: {hit_orig}

---
GOLD (correct) ANSWER
  Index: {gold_index} of {n_options}
  Text:  "{gold_option}"

MODEL'S PREDICTION
  Index: {pred_index} of {n_options}
  Text:  "{pred_option}"
  Result: {result}

---
MODEL'S CHAIN-OF-THOUGHT
{reasoning_trace}

---

Based on the chain-of-thought above, what strategy did the model use?
Consider (but do not limit yourself to):
  - Deriving the answer from the question alone, then explicitly scanning the option list for a match
  - Deriving the answer from the question alone, then selecting by recognition without explicitly iterating the list (no scan phrases in trace)
  - Confidence-ranked shortlisting: generating several candidate answers from prior knowledge, scanning to find which are present, and picking the top-ranked match
  - Sequential scan from the start, stopping early (first plausible option or context/attention limit)
  - Sequential scan from the end, stopping at the last plausible option encountered
  - Exploiting structural cues (block grouping, option density, sparse vs. contiguous layout)
  - Anchoring to a positional heuristic (round index, middle of list, e.g. ~500 or ~250 out of 1000)
  - Surface/keyword matching: picking the option that shares the most words or phrases with the question, without independent derivation
  - Partial elimination of clearly wrong options, then selecting among the remainder
  - Explicit acknowledgement of overload / capitulation to a default guess

Quote specific phrases from the chain-of-thought to support your label.
Respond with JSON only, using exactly this structure:
{{
  "strategy_label": "<2-5 word label, e.g. 'sequential scan from start'>",
  "strategy_description": "<1-2 sentences citing specific phrases from the chain-of-thought>",
  "confidence": "<high | medium | low>"
}}\
"""

SYNTHESIS_TEMPLATE = """\
You are synthesizing open-coded reasoning strategy labels into a coherent taxonomy.

{n_codes} traces from {models_desc} were open-coded by a judge after answering MCQs
with {n_options_range} options. Each record below captures the inferred reasoning strategy.

---
LABEL FREQUENCY TABLE  (exact counts — use these, do not re-estimate)
{label_freq_block}
---

ALL OPEN CODES  (label · description · condition · correctness)
{codes_block}
---

TASK — work in two explicit steps:

Step 1 — Normalize: identify synonym clusters (labels that describe the same underlying
behavior with different wording, e.g. "derive then scan" / "derive answer then scan").
Keep this step in mind before grouping — do not let wording variation split one strategy
into two codes.

Step 2 — Group: collapse synonyms and related variants into a compact, non-overlapping
taxonomy of 5–9 strategies ordered by true frequency (use the table above, not guesses).
For each strategy:
  - Assign a code: S1, S2, … (S1 = most frequent)
  - Give a 2–4 word name
  - Write a 2–3 sentence description
  - Describe the behavioral signature: exact trace phrases or patterns that trigger this code
  - Set frequency from the table: "high" (>30%), "medium" (10–30%), "rare" (<10%)

In the notes field include:
  - Which synonym clusters you collapsed and why
  - Any strategy that appears notably more or less common at specific window positions
    (0.0 = gold near start of list, 1.0 = gold near end) or placement modes (block/sparse)
  - Any behaviors observed in traces that don't fit the taxonomy

Respond with JSON only:
{{
  "taxonomy": [
    {{
      "code": "S1",
      "name": "<2-4 word name>",
      "description": "<2-3 sentence definition>",
      "behavioral_signature": "<trace phrases or patterns that trigger this code>",
      "frequency": "<high | medium | rare>"
    }}
  ],
  "notes": "<synonym clusters collapsed · condition patterns · ambiguities · outliers>"
}}\
"""


def build_open_coding_prompt(ctx: dict) -> str:
    mode = ctx["placement_mode"]
    mode_desc = (
        "the original 10 options form a contiguous block; the block position shifts with the window"
        if mode == "block"
        else "the original 10 options are spread one-per-segment across the full list"
    )
    result_str = "CORRECT" if ctx["correct"] else "INCORRECT"
    hit_orig = str(ctx["hit_original_option"]) if ctx["hit_original_option"] is not None else "unknown"
    reasoning_trace = ctx.get("reasoning_trace") or "[no reasoning trace available]"

    return OPEN_CODING_TEMPLATE.format(
        question=ctx["question"],
        category=ctx["category"],
        option_dep=ctx["option_dep"],
        n_options=ctx["n_options"],
        window=ctx["window"],
        mode=mode,
        mode_desc=mode_desc,
        hit_orig=hit_orig,
        gold_index=ctx["gold_index"],
        gold_option=ctx["gold_option"],
        pred_index=ctx["predicted_index"],
        pred_option=ctx["predicted_option"],
        result=result_str,
        reasoning_trace=reasoning_trace,
    )


def build_synthesis_prompt(all_codes: list, n_options_range: str, models_desc: str) -> str:
    from collections import Counter
    label_counts = Counter(c["strategy_label"] for c in all_codes)
    freq_lines = [
        f"  {count:3d} ({100 * count / len(all_codes):4.1f}%)  {label}"
        for label, count in label_counts.most_common()
    ]
    lines = []
    for i, c in enumerate(all_codes):
        lines.append(
            f"[{i+1:03d}] condition={c['condition']}  correct={c['correct']}\n"
            f"  label:       {c['strategy_label']}\n"
            f"  description: {c['strategy_description']}\n"
            f"  confidence:  {c['confidence']}"
        )
    return SYNTHESIS_TEMPLATE.format(
        n_codes=len(all_codes),
        n_options_range=n_options_range,
        models_desc=models_desc,
        label_freq_block="\n".join(freq_lines),
        codes_block="\n\n".join(lines),
    )


# -------------------------------------------------------------------------
# API calls
# -------------------------------------------------------------------------

def call_open_code(client: Together, model: str, prompt: str) -> Tuple[OpenCode, str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8096,
        temperature=0,
        response_format={
            "type": "json_object",
            "schema": OpenCode.model_json_schema(),
        },
    )
    raw = resp.choices[0].message.content.strip()
    return OpenCode.model_validate_json(raw), raw


def call_synthesis(
    client: Together, model: str, all_codes: list, n_options_range: str, models_desc: str
) -> Tuple[TaxonomySynthesis, str]:
    prompt = build_synthesis_prompt(all_codes, n_options_range, models_desc)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0,
        response_format={
            "type": "json_object",
            "schema": TaxonomySynthesis.model_json_schema(),
        },
    )
    raw = resp.choices[0].message.content.strip()
    return TaxonomySynthesis.model_validate_json(raw), raw


# -------------------------------------------------------------------------
# Markdown rendering
# -------------------------------------------------------------------------

def taxonomy_to_markdown(tax: TaxonomySynthesis) -> str:
    lines = [
        "# Draft Taxonomy of Reasoning Strategies\n",
        "_Auto-generated by pilot_taxonomy_judge.py — validate against human open-coding before freezing._\n",
        "| Code | Name | Frequency | Description |",
        "|------|------|-----------|-------------|",
    ]
    for e in tax.taxonomy:
        desc = e.description.replace("\n", " ").replace("|", "\\|")
        lines.append(f"| **{e.code}** | {e.name} | {e.frequency} | {desc} |")

    lines += ["", "## Behavioral Signatures", ""]
    for e in tax.taxonomy:
        lines.append(f"### {e.code} — {e.name}")
        lines.append(e.behavioral_signature)
        lines.append("")

    if tax.notes:
        lines += ["## Notes", "", tax.notes, ""]

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args):
    client = Together()
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    open_codes_path = out_dir / "open_codes.jsonl"
    all_codes: list = []

    # ---- Stage 1: open coding ----------------------------------------
    print("=" * 60)
    print("STAGE 1: open coding")
    print("=" * 60)

    pool = {m: list(tiers) for m, tiers in MODEL_POOL.items()}
    if args.models:
        unknown = set(args.models) - set(MODEL_POOL)
        if unknown:
            raise SystemExit(f"Unknown models: {sorted(unknown)}. Available: {sorted(MODEL_POOL)}")
        pool = {m: pool[m] for m in args.models}
    if args.n_options:
        pool = {m: [n for n in tiers if n in args.n_options] for m, tiers in pool.items()}
        pool = {m: tiers for m, tiers in pool.items() if tiers}

    for model_slug, n_options_tiers in pool.items():
        for n_options in n_options_tiers:
            for window in WINDOWS:
                for mode in MODES:
                    run_dir = condition_run_dir(model_slug, n_options, window, mode)
                    condition_str = f"{model_slug}__n{n_options}_w{window:.2f}_{mode}"

                    if run_dir is None:
                        print(f"[SKIP] no run dir: {condition_str}")
                        continue

                    preds = load_predictions(run_dir)
                    ds_by_qid = load_dataset_by_qid(run_dir)
                    reasoning_by_qid = load_reasoning_traces(run_dir)

                    if not ds_by_qid:
                        print(f"[SKIP] dataset not found for: {condition_str} (run_dir={run_dir})")
                        continue

                    n_with_reasoning = sum(1 for v in reasoning_by_qid.values() if v)
                    print(f"\n{condition_str}: (run={run_dir.name})  reasoning traces: {n_with_reasoning}/{len(reasoning_by_qid)}")

                    sampled = stratified_sample(preds, args.n_per_condition, args.seed)
                    print(f"  sampled {len(sampled)} traces")

                    for pred in sampled:
                        qid = str(pred["question_id"])
                        ds_row = ds_by_qid.get(qid)
                        if ds_row is None:
                            print(f"  qid={qid}: not in dataset, skipping")
                            continue

                        reasoning = reasoning_by_qid.get(qid, "")
                        ctx = build_context(pred, ds_row, reasoning, model_slug=model_slug)
                        prompt = build_open_coding_prompt(ctx)

                        if args.dry_run:
                            print(f"\n  --- DRY RUN (qid={qid}, {condition_str}) ---")
                            print(prompt[:800] + "\n  ...")
                            continue

                        try:
                            code_obj, raw = call_open_code(client, args.judge_model, prompt)
                            record = {
                                "question_id": qid,
                                "model_slug": model_slug,
                                "condition": condition_str,
                                "n_options": n_options,
                                "window": window,
                                "mode": mode,
                                "correct": pred["correct"],
                                "gold_index": pred["gold_index"],
                                "predicted_index": pred["predicted_index"],
                                "strategy_label": code_obj.strategy_label,
                                "strategy_description": code_obj.strategy_description,
                                "confidence": code_obj.confidence,
                                "raw_response": raw,
                            }
                            with open(open_codes_path, "a") as fout:
                                fout.write(json.dumps(record) + "\n")
                            all_codes.append(record)
                            flag = "✓" if pred["correct"] else "✗"
                            print(f"  {flag} qid={qid}  [{code_obj.confidence}] {code_obj.strategy_label} | {code_obj.strategy_description}")
                        except Exception as exc:
                            print(f"  ERROR qid={qid}: {exc}")

    if args.dry_run:
        print("\nDry run complete — no API calls made.")
        return

    n_coded = len(all_codes)
    print(f"\nStage 1 complete: {n_coded} traces coded → {open_codes_path}")

    if n_coded < 5:
        print("Too few traces coded — skipping synthesis.")
        return

    # ---- Stage 2: synthesis ------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: taxonomy synthesis")
    print("=" * 60)

    n_options_range = "+".join(
        str(n) for n in sorted(set(n for tiers in pool.values() for n in tiers))
    )
    models_desc = ", ".join(pool)
    try:
        tax, _ = call_synthesis(client, args.judge_model, all_codes, n_options_range, models_desc)

        tax_json_path = out_dir / "taxonomy_draft.json"
        tax_md_path = out_dir / "taxonomy_draft.md"

        with open(tax_json_path, "w") as f:
            json.dump(tax.model_dump(), f, indent=2)

        md = taxonomy_to_markdown(tax)
        with open(tax_md_path, "w") as f:
            f.write(md)

        print(f"\nTaxonomy → {tax_json_path}")
        print(f"Markdown → {tax_md_path}")
        print("\n" + "=" * 60)
        print(md)

    except Exception as exc:
        print(f"Synthesis failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Open-code reasoning strategies from gpt-oss-120b CoT traces and synthesize a draft taxonomy."
    )
    parser.add_argument(
        "--n-per-condition",
        type=int,
        default=3,
        help="Traces to sample per (model × n_options × window × mode) condition.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        metavar="SLUG",
        help="Filter to specific model slugs from MODEL_POOL (default: all models).",
    )
    parser.add_argument(
        "--n-options",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Filter to specific n_options tiers across all models (default: per-model pool).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-oss-120B",
        help="Together AI model used as judge.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/taxonomy_pilot",
        help="Base output directory (a timestamp subdirectory is created inside).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts for the first condition without making any API calls.",
    )
    args = parser.parse_args()
    main(args)
