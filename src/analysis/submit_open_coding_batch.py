"""
Build and submit a Together AI batch job for open-coding reasoning strategies.

For every (model × n_options × window × mode) condition in MODEL_POOL, loads
predictions and reasoning traces, stratified-samples n_per_condition traces,
builds an open-coding prompt for each, and submits all prompts as a single
Together AI batch job.

A manifest.json is saved alongside the JSONL so retrieve_open_coding_batch.py
can reconstruct full records without re-loading datasets.

Rough cost at default settings (n=25, Llama-3.3-70B-Instruct-Turbo judge):
  30 conditions × 25 samples = 750 calls
  ~4K input tokens avg → 3M tokens × $0.59/M ≈ $1.80 input
  ~250 output tokens avg            → $0.11 output
  Total: ~$2

Usage:
    python preprocess/analysis/submit_open_coding_batch.py
    python preprocess/analysis/submit_open_coding_batch.py \\
        --n-per-condition 25 \\
        --judge-model meta-llama/Llama-3.3-70B-Instruct-Turbo \\
        --dry-run
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from together import Together

sys.path.insert(0, str(Path(__file__).parent))
from pilot_taxonomy_judge import (
    MODEL_POOL, WINDOWS, MODES,
    OpenCode,
    build_context, build_open_coding_prompt,
    condition_run_dir, load_dataset_by_qid,
    load_predictions, load_reasoning_traces,
    stratified_sample,
)

load_dotenv()

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_N_PER_CONDITION = 25
DEFAULT_OUTPUT_DIR = "out/open_coding_batches"


def build_batch_request(custom_id: str, prompt: str, model: str) -> dict:
    return {
        "custom_id": custom_id,
        "body": {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "response_format": {
                "type": "json_object",
                "schema": OpenCode.model_json_schema(),
            },
        },
    }


def main(args):
    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    pool = dict(MODEL_POOL)
    if args.models:
        unknown = set(args.models) - set(MODEL_POOL)
        if unknown:
            raise SystemExit(f"Unknown models: {sorted(unknown)}. Available: {sorted(MODEL_POOL)}")
        pool = {m: pool[m] for m in args.models}
    if args.n_options:
        pool = {m: [n for n in tiers if n in args.n_options] for m, tiers in pool.items()}
        pool = {m: tiers for m, tiers in pool.items() if tiers}

    requests = []
    manifest: dict = {}  # custom_id → metadata needed by retrieve script

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
                    print(f"{condition_str}: {len(sampled)} traces sampled")

                    for pred in sampled:
                        qid = str(pred["question_id"])
                        ds_row = ds_by_qid.get(qid)
                        if ds_row is None:
                            continue

                        reasoning = reasoning_by_qid.get(qid, "")
                        ctx = build_context(pred, ds_row, reasoning, model_slug=model_slug)
                        prompt = build_open_coding_prompt(ctx)

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
                        }

    if args.dry_run:
        print("\n[DRY RUN] First prompt preview (first 1200 chars):\n")
        if requests:
            print(requests[0]["body"]["messages"][0]["content"][:1200])
            print("\n...")
        return

    if not requests:
        raise SystemExit("No requests to submit — all conditions skipped.")

    # ---- Confirmation banner ------------------------------------------------
    from collections import Counter
    per_model = Counter(m["model_slug"] for m in manifest.values())
    n_conditions = len(set(m["condition"] for m in manifest.values()))
    n_total = len(requests)
    avg_input_tokens = 4000
    avg_output_tokens = 250
    est_input_m  = (n_total * avg_input_tokens)  / 1_000_000
    est_output_m = (n_total * avg_output_tokens) / 1_000_000

    print("\n" + "=" * 60)
    print("  BATCH SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"  Judge model      : {args.judge_model}")
    print(f"  Samples/condition: {args.n_per_condition}")
    print(f"  Conditions       : {n_conditions}")
    print(f"  Total requests   : {n_total}")
    print(f"  Per model        :")
    for slug, count in sorted(per_model.items()):
        print(f"    {slug:<25} {count} requests")
    print(f"  Est. input tokens: ~{est_input_m:.1f}M  (avg {avg_input_tokens} tok/req)")
    print(f"  Est. output tokens: ~{est_output_m:.1f}M (avg {avg_output_tokens} tok/req)")
    print(f"  Output dir       : {out_dir}")
    print("=" * 60)
    answer = input("  Submit? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return
    print()
    # -------------------------------------------------------------------------

    # Save manifest and run config
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)

    config = {
        "judge_model": args.judge_model,
        "n_per_condition": args.n_per_condition,
        "seed": args.seed,
        "pool": {m: list(tiers) for m, tiers in pool.items()},
        "n_requests": len(requests),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write JSONL
    jsonl_path = out_dir / "batch_requests.jsonl"
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Upload and submit
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    print(f"\nUploading {len(requests)} requests...")
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
        f"python preprocess/analysis/retrieve_open_coding_batch.py "
        f"--output-dir {out_dir}"
    )
    with open(out_dir / "retrieve_command.txt", "w") as f:
        f.write(retrieve_cmd + "\n")

    print(f"\nRetrieve results with:\n  {retrieve_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit Together AI batch job for open-coding reasoning strategies."
    )
    parser.add_argument(
        "--n-per-condition",
        type=int,
        default=DEFAULT_N_PER_CONDITION,
        help=f"Traces per (model × n_options × window × mode) condition (default: {DEFAULT_N_PER_CONDITION}).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        metavar="SLUG",
        help="Filter to specific model slugs from MODEL_POOL (default: all).",
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
        default=DEFAULT_JUDGE_MODEL,
        help=f"Together AI model used as judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview first prompt without submitting.",
    )
    args = parser.parse_args()
    main(args)
