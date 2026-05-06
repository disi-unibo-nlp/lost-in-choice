"""
Submit a Together AI batch job to classify reasoning traces against the frozen taxonomy.

Reads the frozen taxonomy (taxonomy_draft.json from synthesize_taxonomy.py) and
auto-selects few-shot exemplars from open_codes.jsonl via keyword similarity between
each open-code description and each taxonomy entry's behavioral_signature.

Output schema per trace:
    {"primary_strategy": "S1", "secondary_strategy": null, "confidence": "high", "rationale": "..."}

Defaults: n=50 per (model × n_options × window × mode), Llama-3.3-70B judge, 3 exemplars/code.

Usage:
    python preprocess/analysis/submit_classification_batch.py \\
        --taxonomy-path out/open_coding_batches/2026-04-30_11-10-35/taxonomy_draft.json \\
        --open-codes-path out/open_coding_batches/2026-04-30_11-10-35/open_codes.jsonl

    # zero-shot (no exemplars):
    python preprocess/analysis/submit_classification_batch.py \\
        --taxonomy-path ... --n-exemplars 0

    # dry-run preview:
    python preprocess/analysis/submit_classification_batch.py \\
        --taxonomy-path ... --open-codes-path ... --dry-run
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from together import Together

sys.path.insert(0, str(Path(__file__).parent))
from pilot_taxonomy_judge import (
    MODEL_POOL, WINDOWS, MODES,
    Classification,
    build_context,
    condition_run_dir, load_dataset_by_qid,
    load_predictions, load_reasoning_traces,
    stratified_sample,
)

load_dotenv()

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_N_PER_CONDITION = 50
DEFAULT_N_EXEMPLARS = 3
DEFAULT_OUTPUT_DIR = "out/classification_batches"


# -------------------------------------------------------------------------
# Prompt construction
# -------------------------------------------------------------------------

CLASSIFICATION_TEMPLATE = """\
You are a judge classifying a reasoning model's chain-of-thought into ONE strategy from a frozen taxonomy.

TAXONOMY  (choose exactly one primary code; one optional secondary if two strategies are clearly sequential)
{taxonomy_block}

---
LABELLED EXAMPLES
{exemplars_block}

---
TARGET TRACE
  Question      : {question}
  Category      : {category}
  Option type   : {option_dep}
  Total options : {n_options}
  Window        : {window}  (0.0 = gold near start, 1.0 = gold near end)
  Mode          : {mode}
  Hit original  : {hit_orig}
  Gold answer   : index {gold_index} — "{gold_option}"
  Model answer  : index {pred_index} — "{pred_option}"
  Result        : {result}

---
MODEL'S CHAIN-OF-THOUGHT
{reasoning_trace}

---
Assign ONE primary strategy code. Set secondary_strategy only if a second distinct
strategy clearly follows the first (e.g. S7 → S8). Quote 1-2 phrases from the
chain-of-thought in your rationale.

Respond with JSON only:
{{
  "primary_strategy": "<code>",
  "secondary_strategy": "<code or null>",
  "confidence": "<high | medium | low>",
  "rationale": "<1-2 sentences citing specific trace phrases>"
}}\
"""


def format_taxonomy_block(taxonomy: list) -> str:
    lines = []
    for entry in taxonomy:
        lines.append(
            f"  {entry['code']} — {entry['name']}\n"
            f"    {entry['description']}\n"
            f"    Signature: {entry['behavioral_signature']}"
        )
    return "\n\n".join(lines)


def select_exemplars(open_codes: list, taxonomy: list, n_per_code: int) -> dict[str, list]:
    """Keyword similarity between open-code description+label and taxonomy behavioral_signature."""
    if n_per_code == 0:
        return {}

    exemplars: dict[str, list] = defaultdict(list)
    for entry in taxonomy:
        code = entry["code"]
        sig_words = set(entry["behavioral_signature"].lower().split())

        scored = []
        for oc in open_codes:
            if oc.get("confidence") != "high":
                continue
            desc_words = set(oc.get("strategy_description", "").lower().split())
            label_words = set(oc.get("strategy_label", "").lower().split())
            score = len((desc_words | label_words) & sig_words) / max(len(sig_words), 1)
            scored.append((score, oc))

        scored.sort(key=lambda x: x[0], reverse=True)
        exemplars[code] = [oc for _, oc in scored[:n_per_code]]

    return dict(exemplars)


def format_exemplars_block(exemplars: dict) -> str:
    if not exemplars:
        return "  (none — classify from taxonomy definition only)"

    lines = []
    for code, examples in exemplars.items():
        for ex in examples:
            lines.append(
                f"  → {code}\n"
                f'    "{ex["strategy_description"]}"'
            )
    return "\n\n".join(lines)


def build_classification_prompt(ctx: dict, taxonomy_block: str, exemplars_block: str) -> str:
    mode = ctx["placement_mode"]
    result_str = "CORRECT" if ctx["correct"] else "INCORRECT"
    hit_orig = str(ctx["hit_original_option"]) if ctx["hit_original_option"] is not None else "unknown"
    reasoning_trace = ctx.get("reasoning_trace") or "[no reasoning trace available]"

    return CLASSIFICATION_TEMPLATE.format(
        taxonomy_block=taxonomy_block,
        exemplars_block=exemplars_block,
        question=ctx["question"],
        category=ctx["category"],
        option_dep=ctx["option_dep"],
        n_options=ctx["n_options"],
        window=ctx["window"],
        mode=mode,
        hit_orig=hit_orig,
        gold_index=ctx["gold_index"],
        gold_option=ctx["gold_option"],
        pred_index=ctx["predicted_index"],
        pred_option=ctx["predicted_option"],
        result=result_str,
        reasoning_trace=reasoning_trace,
    )


def build_batch_request(custom_id: str, prompt: str, model: str) -> dict:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32000,
        "response_format": {
            "type": "json_object",
            "schema": Classification.model_json_schema(),
        },
    }
    if "gpt-oss" in model.lower():
        body["reasoning_effort"] = "low"
    return {"custom_id": custom_id, "body": body}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args):
    # Load taxonomy
    taxonomy_path = Path(args.taxonomy_path)
    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy not found: {taxonomy_path}")
    with open(taxonomy_path) as f:
        tax_data = json.load(f)
    taxonomy = tax_data["taxonomy"]
    valid_codes = {e["code"] for e in taxonomy}
    print(f"Taxonomy: {len(taxonomy)} codes — {', '.join(valid_codes)}")

    # Load open codes for exemplar selection
    open_codes = []
    if args.open_codes_path and Path(args.open_codes_path).exists():
        with open(args.open_codes_path) as f:
            open_codes = [json.loads(line) for line in f if line.strip()]
        print(f"Open codes loaded: {len(open_codes)} traces (for exemplar selection)")
    elif args.n_exemplars > 0:
        print("WARNING: --open-codes-path not given; falling back to zero-shot.")
        args.n_exemplars = 0

    taxonomy_block = format_taxonomy_block(taxonomy)
    exemplars = select_exemplars(open_codes, taxonomy, args.n_exemplars)
    exemplars_block = format_exemplars_block(exemplars)

    n_exemplars_found = sum(len(v) for v in exemplars.values())
    print(f"Exemplars selected: {n_exemplars_found} total ({args.n_exemplars} requested/code)")

    # Build pool
    pool = dict(MODEL_POOL)
    if args.models:
        unknown = set(args.models) - set(MODEL_POOL)
        if unknown:
            raise SystemExit(f"Unknown models: {sorted(unknown)}. Available: {sorted(MODEL_POOL)}")
        pool = {m: pool[m] for m in args.models}
    if args.n_options:
        pool = {m: [n for n in tiers if n in args.n_options] for m, tiers in pool.items()}
        pool = {m: tiers for m, tiers in pool.items() if tiers}

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    requests = []
    manifest: dict = {}

    for model_slug, n_options_tiers in pool.items():
        for n_options in n_options_tiers:
            for window in WINDOWS:
                for mode in MODES:
                    run_dir = condition_run_dir(model_slug, n_options, window, mode)
                    condition_str = f"{model_slug}__n{n_options}_w{window:.2f}_{mode}"

                    if run_dir is None:
                        print(f"[SKIP] no completed run: {condition_str}")
                        continue

                    preds = load_predictions(run_dir)
                    ds_by_qid = load_dataset_by_qid(run_dir)
                    reasoning_by_qid = load_reasoning_traces(run_dir)

                    if not ds_by_qid:
                        print(f"[SKIP] dataset not found: {condition_str}")
                        continue

                    sampled = stratified_sample(preds, args.n_per_condition, args.seed)
                    print(f"{condition_str}: {len(sampled)} traces")

                    for pred in sampled:
                        qid = str(pred["question_id"])
                        ds_row = ds_by_qid.get(qid)
                        if ds_row is None:
                            continue

                        reasoning = reasoning_by_qid.get(qid, "")
                        ctx = build_context(pred, ds_row, reasoning, model_slug=model_slug)
                        prompt = build_classification_prompt(ctx, taxonomy_block, exemplars_block)

                        custom_id = f"{condition_str}__{qid}"
                        requests.append(build_batch_request(custom_id, prompt, args.judge_model))
                        manifest[custom_id] = {
                            "model_slug": model_slug,
                            "condition": condition_str,
                            "n_options": n_options,
                            "window": window,
                            "mode": mode,
                            "question_id": qid,
                            "correct": pred["correct"],
                            "gold_index": pred["gold_index"],
                            "predicted_index": pred["predicted_index"],
                            "option_dep": ds_row.get("option_dependent_vs_independent", "unknown"),
                        }

    if args.dry_run:
        print(f"\n[DRY RUN] {len(requests)} requests would be submitted.")
        if requests:
            print("\nFirst prompt preview (first 2000 chars):\n")
            print(requests[0]["body"]["messages"][0]["content"][:2000])
            print("\n...")
        return

    if not requests:
        raise SystemExit("No requests to submit — all conditions skipped.")

    # ---- Confirmation banner --------------------------------------------
    from collections import Counter
    per_model = Counter(m["model_slug"] for m in manifest.values())
    n_conditions = len(set(m["condition"] for m in manifest.values()))
    n_total = len(requests)
    est_input_m  = (n_total * 4500) / 1_000_000
    est_output_m = (n_total * 300)  / 1_000_000

    print("\n" + "=" * 60)
    print("  CLASSIFICATION BATCH SUMMARY")
    print("=" * 60)
    print(f"  Judge model      : {args.judge_model}")
    print(f"  Taxonomy codes   : {', '.join(valid_codes)}")
    print(f"  Exemplars / code : {args.n_exemplars}")
    print(f"  Samples/condition: {args.n_per_condition}")
    print(f"  Conditions       : {n_conditions}")
    print(f"  Total requests   : {n_total}")
    print(f"  Per model        :")
    for slug, count in sorted(per_model.items()):
        print(f"    {slug:<28} {count} requests")
    print(f"  Est. input tokens : ~{est_input_m:.1f}M")
    print(f"  Est. output tokens: ~{est_output_m:.1f}M")
    print(f"  Output dir        : {out_dir}")
    print("=" * 60)
    answer = input("  Submit? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return
    print()
    # ---------------------------------------------------------------------

    # Save manifest, config, exemplars used
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    config = {
        "judge_model": args.judge_model,
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_codes": list(valid_codes),
        "n_exemplars_per_code": args.n_exemplars,
        "n_per_condition": args.n_per_condition,
        "seed": args.seed,
        "pool": {m: list(tiers) for m, tiers in pool.items()},
        "n_requests": n_total,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save the prompt template that was used (for reproducibility)
    with open(out_dir / "exemplars_used.json", "w") as f:
        json.dump({k: [e["strategy_description"] for e in v] for k, v in exemplars.items()}, f, indent=2)

    jsonl_path = out_dir / "batch_requests.jsonl"
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    print(f"Uploading {n_total} requests...")
    file_resp = client.files.upload(file=str(jsonl_path), purpose="batch-api", check=False)
    print(f"File uploaded: {file_resp.id}")

    batch = client.batches.create(
        input_file_id=file_resp.id,
        endpoint="/v1/chat/completions",
    )
    job_id = batch.job.id
    print(f"Batch job created: {job_id}")
    print(f"Status: {client.batches.retrieve(job_id).status}")

    with open(out_dir / "job_id.txt", "w") as f:
        f.write(job_id)

    retrieve_cmd = (
        f"python preprocess/analysis/retrieve_classification_batch.py "
        f"--output-dir {out_dir}"
    )
    with open(out_dir / "retrieve_command.txt", "w") as f:
        f.write(retrieve_cmd + "\n")

    print(f"\nRetrieve results with:\n  {retrieve_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit Together AI batch job for strategy classification against frozen taxonomy."
    )
    parser.add_argument(
        "--taxonomy-path", required=True,
        help="Path to taxonomy_draft.json produced by synthesize_taxonomy.py.",
    )
    parser.add_argument(
        "--open-codes-path", default=None,
        help="Path to open_codes.jsonl for auto exemplar selection (optional).",
    )
    parser.add_argument(
        "--n-exemplars", type=int, default=DEFAULT_N_EXEMPLARS,
        help=f"Exemplars per taxonomy code selected from open_codes.jsonl (default: {DEFAULT_N_EXEMPLARS}; 0 = zero-shot).",
    )
    parser.add_argument(
        "--n-per-condition", type=int, default=DEFAULT_N_PER_CONDITION,
        help=f"Traces per (model × n_options × window × mode) condition (default: {DEFAULT_N_PER_CONDITION}).",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None, metavar="SLUG",
        help="Filter to specific model slugs from MODEL_POOL.",
    )
    parser.add_argument(
        "--n-options", type=int, nargs="+", default=None, metavar="N",
        help="Filter to specific n_options tiers.",
    )
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Together AI judge model (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview first prompt without submitting.",
    )
    args = parser.parse_args()
    main(args)
