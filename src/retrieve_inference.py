"""
Retrieve results from a completed Together AI batch inference job and
parse structured JSON answers into an evaluated dataset.

Usage:
    python src/retrieve_inference.py \\
        --output-dir out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__window_0.50__sparse/2026-04-16_12-00-00 \\
        --job-id <BATCH_JOB_ID>

    # If job-id is already saved in output_dir/job_id.txt, it is picked up automatically:
    python src/retrieve_inference.py \\
        --output-dir out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__window_0.50__sparse/2026-04-16_12-00-00

Outputs (written into --output-dir):
    raw_completions.jsonl   — raw API response lines as downloaded
    predictions.jsonl       — one JSON record per question with parsed answer
    metrics.json            — accuracy + per-category breakdown
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Optional

from dotenv import load_dotenv
from together import Together

load_dotenv()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_answer(response_body: dict) -> Optional[int]:
    """
    Extract answer_index from a structured JSON response.

    The model was forced via json_schema response_format to emit
        {"answer_index": <int>}
    so the happy path is a direct JSON parse of the content field.
    Falls back to scanning the content string for the first integer
    if the schema was somehow bypassed.
    """
    try:
        choices = response_body["choices"]
        if not choices:
            return None

        content = choices[0]["message"].get("content", "")
        if not content:
            return None

        # Primary path: structured JSON
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "answer_index" in parsed:
            val = parsed["answer_index"]
            if isinstance(val, int):
                return val
            # Sometimes serialised as string "42"
            if isinstance(val, str) and val.isdigit():
                return int(val)

    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass

    # Fallback: scan content for first integer token
    import re
    content_str = str(response_body.get("choices", [{}])[0].get("message", {}).get("content", ""))
    match = re.search(r"\b(\d+)\b", content_str)
    if match:
        return int(match.group(1))

    return None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_results(job_id: str, output_dir: str, client: Together) -> str:
    """Download raw completions; return path to the saved JSONL file."""
    print(f"Checking batch status for job: {job_id}")
    batch = client.batches.retrieve(job_id)
    print(f"  Status: {batch.status}")

    if batch.status != "COMPLETED":
        print(
            "Batch is not yet completed. Re-run this script once the job finishes.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_path = os.path.join(output_dir, "raw_completions.jsonl")
    print(f"Downloading results to: {raw_path}")
    with client.files.with_streaming_response.content(id=batch.output_file_id) as resp:
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)

    print("Download complete.")
    return raw_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(raw_path: str, output_dir: str, run_config: dict):
    n_options = run_config.get("n_options", 1000)

    print("\nParsing completions...")
    with open(raw_path) as f:
        completions = [json.loads(line) for line in f if line.strip()]

    print(f"  {len(completions)} completion lines loaded")

    # Load the original dataset to retrieve gold labels and metadata
    dataset_path = run_config.get("dataset_path")
    gold_map: dict = {}  # question_id → {answer_index, category, window, mode, ...}
    is_baseline = run_config.get("baseline", False)
    if dataset_path and os.path.isdir(dataset_path):
        from datasets import load_from_disk
        ds = load_from_disk(dataset_path)
        answer_index_key = "answer_index" if is_baseline else "new_answer_index"
        for row in ds:
            qid = str(row["question_id"])
            # For augmented datasets, map which indices in new_options are original options
            if not is_baseline and "new_options" in row and "options" in row:
                orig_set = set(row["options"])
                orig_indices = frozenset(
                    i for i, opt in enumerate(row["new_options"]) if opt in orig_set
                )
            else:
                orig_indices = None
            gold_map[qid] = {
                "gold_index": row[answer_index_key],
                "category": row.get("category", "unknown"),
                "option_dependent_vs_independent": row.get("option_dependent_vs_independent", "unknown"),
                "window": row.get("window"),
                "placement_mode": row.get("placement_mode"),
                "original_option_indices": orig_indices,
            }
    else:
        print("  WARNING: dataset_path not found — skipping gold-label lookup.")

    predictions = []
    parse_failures = 0

    for item in completions:
        qid = item.get("custom_id", "")
        response_body = item.get("response", {}).get("body", {})

        predicted = parse_answer(response_body)

        if predicted is None:
            parse_failures += 1

        meta = gold_map.get(qid, {})
        gold = meta.get("gold_index")
        correct = (predicted == gold) if (predicted is not None and gold is not None) else None

        orig_indices = meta.get("original_option_indices")
        if predicted is not None and orig_indices is not None:
            hit_original_option = predicted in orig_indices
        else:
            hit_original_option = None  # not applicable (baseline) or unparsed

        record = {
            "question_id": qid,
            "predicted_index": predicted,
            "gold_index": gold,
            "correct": correct,
            "hit_original_option": hit_original_option,
            "category": meta.get("category", "unknown"),
            "option_dependent_vs_independent": meta.get("option_dependent_vs_independent", "unknown"),
            "window": meta.get("window"),
            "placement_mode": meta.get("placement_mode"),
            "parse_failed": predicted is None,
        }
        predictions.append(record)

    # Save predictions
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")
    print(f"Predictions saved to: {pred_path}")

    # --- Metrics ---
    evaluated = [r for r in predictions if r["correct"] is not None]
    n_total = len(predictions)
    n_correct = sum(1 for r in evaluated if r["correct"])
    accuracy = n_correct / len(evaluated) if evaluated else 0.0

    # Per-category
    cat_totals: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in evaluated:
        cat = r["category"]
        cat_totals[cat]["total"] += 1
        if r["correct"]:
            cat_totals[cat]["correct"] += 1

    # Per dependency label
    dep_totals: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in evaluated:
        dep = r["option_dependent_vs_independent"]
        dep_totals[dep]["total"] += 1
        if r["correct"]:
            dep_totals[dep]["correct"] += 1

    # Per dependency × category
    dep_cat_totals: dict = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for r in evaluated:
        dep = r["option_dependent_vs_independent"]
        cat = r["category"]
        dep_cat_totals[dep][cat]["total"] += 1
        if r["correct"]:
            dep_cat_totals[dep][cat]["correct"] += 1

    # Predicted index distribution (detect positional bias) — always 10 bins
    pred_indices = [r["predicted_index"] for r in predictions if r["predicted_index"] is not None]
    seg_size = max(1, n_options // 10)
    segment_counts: dict = Counter(idx // seg_size for idx in pred_indices)

    # Option-type breakdown: original 10 vs sampled distractors
    typed = [r for r in predictions if r["hit_original_option"] is not None]
    n_unparsed = sum(1 for r in predictions if r["parse_failed"])
    n_correct_original = sum(1 for r in typed if r["hit_original_option"] and r["correct"])
    n_wrong_original   = sum(1 for r in typed if r["hit_original_option"] and not r["correct"])
    n_distractor       = sum(1 for r in typed if not r["hit_original_option"])
    n_typed = len(typed)
    option_type_stats: Optional[dict] = None
    if n_typed > 0:
        def pct(n): return round(100 * n / n_typed, 2)
        option_type_stats = {
            "n_parsed": n_typed,
            "n_unparsed": n_unparsed,
            "n_correct_original": n_correct_original,
            "n_wrong_original": n_wrong_original,
            "n_any_original": n_correct_original + n_wrong_original,
            "n_distractor": n_distractor,
            "pct_correct_original": pct(n_correct_original),
            "pct_wrong_original": pct(n_wrong_original),
            "pct_any_original": pct(n_correct_original + n_wrong_original),
            "pct_distractor": pct(n_distractor),
        }

    metrics = {
        "model": run_config.get("model_name"),
        "window": run_config.get("window"),
        "placement_mode": run_config.get("mode"),
        "n_options": n_options,
        "n_questions": n_total,
        "n_evaluated": len(evaluated),
        "n_correct": n_correct,
        "accuracy": round(accuracy, 4),
        "parse_failures": parse_failures,
        "per_category": {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            }
            for cat, v in sorted(cat_totals.items())
        },
        "per_dependency_label": {
            dep: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
            }
            for dep, v in sorted(dep_totals.items())
        },
        "per_dependency_per_category": {
            dep: {
                cat: {
                    "correct": cv["correct"],
                    "total": cv["total"],
                    "accuracy": round(cv["correct"] / cv["total"], 4) if cv["total"] else 0.0,
                }
                for cat, cv in sorted(cat_map.items())
            }
            for dep, cat_map in sorted(dep_cat_totals.items())
        },
        "predicted_index_by_segment": {
            f"seg_{s*seg_size}_{s*seg_size + seg_size - 1}": segment_counts.get(s, 0)
            for s in range(10)
        },
        "option_type_stats": option_type_stats,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*52}")
    print(f"  Model   : {metrics['model']}")
    print(f"  Window  : {metrics['window']}  |  Mode: {metrics['placement_mode']}")
    print(f"  Accuracy: {n_correct}/{len(evaluated)}  ({100*accuracy:.1f}%)")
    print(f"  Parse failures: {parse_failures}")
    print(f"\n  Per-category accuracy:")
    for cat, v in metrics["per_category"].items():
        print(f"    {cat:<22} {v['correct']:>4}/{v['total']:<4}  ({100*v['accuracy']:.1f}%)")
    print(f"\n  Per-dependency accuracy:")
    for dep, v in metrics["per_dependency_label"].items():
        print(f"    {dep:<30} {v['correct']:>4}/{v['total']:<4}  ({100*v['accuracy']:.1f}%)")
    print(f"\n  Per-dependency per-category accuracy:")
    for dep, cat_map in metrics["per_dependency_per_category"].items():
        print(f"    [{dep}]")
        for cat, v in cat_map.items():
            print(f"      {cat:<22} {v['correct']:>4}/{v['total']:<4}  ({100*v['accuracy']:.1f}%)")
    print(f"\n  Predicted-index segment distribution:")
    for seg_label, count in metrics["predicted_index_by_segment"].items():
        bar = "█" * (count * 40 // max(segment_counts.values(), default=1))
        print(f"    {seg_label:<16} {count:>5}  {bar}")

    if option_type_stats:
        ots = option_type_stats
        print(f"\n  Option-type breakdown (of {ots['n_parsed']} parsed predictions):")
        print(f"    Unparsed (None)     : {ots['n_unparsed']}")
        print(f"    ├─ correct original : {ots['n_correct_original']:>5}  ({ots['pct_correct_original']:.1f}%)  ← gold (one of original 10, correct)")
        print(f"    ├─ wrong original   : {ots['n_wrong_original']:>5}  ({ots['pct_wrong_original']:.1f}%)  ← one of the other 9 original options")
        print(f"    └─ distractor       : {ots['n_distractor']:>5}  ({ots['pct_distractor']:.1f}%)  ← sampled distractor")
        print(f"    Any original        : {ots['n_any_original']:>5}  ({ots['pct_any_original']:.1f}%)")

    print(f"{'='*52}")
    print(f"\nMetrics saved to: {metrics_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        print(f"ERROR: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve job_id
    job_id = args.job_id
    if not job_id:
        id_file = os.path.join(output_dir, "job_id.txt")
        if os.path.isfile(id_file):
            with open(id_file) as f:
                job_id = f.read().strip()
            print(f"Loaded job ID from {id_file}: {job_id}")
        else:
            print(
                "ERROR: --job-id not provided and job_id.txt not found in output_dir.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load run config for dataset path and metadata
    config_path = os.path.join(output_dir, "run_config.json")
    run_config: dict = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            run_config = json.load(f)
        print(f"Loaded run config: {config_path}")
    else:
        print("WARNING: run_config.json not found — metadata will be limited.")

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in environment. Check your .env file.")

    client = Together(api_key=TOGETHER_API_KEY)

    raw_path = retrieve_results(job_id, output_dir, client)
    evaluate(raw_path, output_dir, run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve and evaluate a completed Together AI batch inference job."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory written by run_inference.py (contains run_config.json and job_id.txt).",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Together AI batch job ID. If omitted, read from job_id.txt in --output-dir.",
    )
    args = parser.parse_args()
    main(args)
