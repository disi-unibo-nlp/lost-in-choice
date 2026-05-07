"""
Local vLLM inference on one augmented MMLU-Pro configuration.

Usage:
    python src/run_inference_vllm.py \\
        --window 0.5 \\
        --mode sparse \\
        --model-name Qwen/Qwen3-27B-FP8 \\
        --limit 10          # omit for full run

Dataset path resolved automatically from --window and --mode:
    data/bench/mmlu-pro-augmented-clean-<n_options>/seed_<seed>/window_<w:.2f>/<mode>/

Structured decoding:
    Every request uses vLLM guided JSON decoding (outlines backend) with schema
        {"answer_index": <integer 0 … N_OPTIONS-1>}
    so the model is forced to emit a valid integer index.

For Qwen3 models, thinking mode is enabled by default via the chat template
(pass --no-thinking to disable it).

Outputs (under out/completions/<model>/<dataset_tag>/<timestamp>/):
    raw_completions.jsonl   — one record per question in Together-compatible format
    predictions.jsonl       — parsed answer per question with gold labels
    metrics.json            — accuracy + per-category breakdown
    run_config.json         — full run configuration for reproducibility
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

from datasets import load_from_disk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_WINDOWS = [0.0, 0.25, 0.5, 0.75, 1.0]
VALID_MODES = ["block", "sparse"]
DEFAULT_MODEL = "Qwen/Qwen3-27B-FP8"
DEFAULT_MAX_TOKENS = 32000


# ---------------------------------------------------------------------------
# JSON schema for structured answer
# ---------------------------------------------------------------------------

def build_json_schema(n_options: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "answer_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": n_options - 1,
                "description": f"Index (0–{n_options - 1}) of the correct option.",
            }
        },
        "required": ["answer_index"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_messages(
    item: dict,
    n_options: int,
    options_key: str = "new_options",
) -> Tuple[str, list]:
    """Return (custom_id, messages_list) for one dataset row."""
    qid = str(item["question_id"])
    question = item["question"]
    options: List[str] = item[options_key]

    lines = [f"Question: {question}", "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"({i}) {opt}")
    lines += [
        "",
        f"Select the single correct option. "
        f"Respond with JSON: {{\"answer_index\": <integer from 0 to {n_options - 1}>}}",
    ]

    return qid, [{"role": "user", "content": "\n".join(lines)}]


# ---------------------------------------------------------------------------
# Answer parsing (identical logic to retrieve_inference.py)
# ---------------------------------------------------------------------------

def parse_answer(content: str) -> Optional[int]:
    """Extract answer_index from a JSON content string."""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "answer_index" in parsed:
            val = parsed["answer_index"]
            if isinstance(val, int):
                return val
            if isinstance(val, str) and val.isdigit():
                return int(val)
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    match = re.search(r"\b(\d+)\b", content)
    if match:
        return int(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Evaluation (mirrors retrieve_inference.py::evaluate exactly)
# ---------------------------------------------------------------------------

def evaluate(raw_path: str, output_dir: str, run_config: dict):
    n_options = run_config.get("n_options", 1000)

    print("\nParsing completions...")
    with open(raw_path) as f:
        completions = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(completions)} completion lines loaded")

    dataset_path = run_config.get("dataset_path")
    gold_map: dict = {}
    is_baseline = run_config.get("baseline", False)
    if dataset_path and os.path.isdir(dataset_path):
        ds = load_from_disk(dataset_path)
        answer_index_key = "answer_index" if is_baseline else "new_answer_index"
        for row in ds:
            qid = str(row["question_id"])
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
        content = (
            item.get("response", {})
            .get("body", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        predicted = parse_answer(content)

        if predicted is None:
            parse_failures += 1

        meta = gold_map.get(qid, {})
        gold = meta.get("gold_index")
        correct = (predicted == gold) if (predicted is not None and gold is not None) else None

        orig_indices = meta.get("original_option_indices")
        hit_original_option = (
            (predicted in orig_indices)
            if (predicted is not None and orig_indices is not None)
            else None
        )

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

    pred_path = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_path, "w") as f:
        for rec in predictions:
            f.write(json.dumps(rec) + "\n")
    print(f"Predictions saved to: {pred_path}")

    evaluated = [r for r in predictions if r["correct"] is not None]
    n_total = len(predictions)
    n_correct = sum(1 for r in evaluated if r["correct"])
    accuracy = n_correct / len(evaluated) if evaluated else 0.0

    cat_totals: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in evaluated:
        cat_totals[r["category"]]["total"] += 1
        if r["correct"]:
            cat_totals[r["category"]]["correct"] += 1

    dep_totals: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in evaluated:
        dep_totals[r["option_dependent_vs_independent"]]["total"] += 1
        if r["correct"]:
            dep_totals[r["option_dependent_vs_independent"]]["correct"] += 1

    dep_cat_totals: dict = defaultdict(
        lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    for r in evaluated:
        dep_cat_totals[r["option_dependent_vs_independent"]][r["category"]]["total"] += 1
        if r["correct"]:
            dep_cat_totals[r["option_dependent_vs_independent"]][r["category"]]["correct"] += 1

    pred_indices = [r["predicted_index"] for r in predictions if r["predicted_index"] is not None]
    seg_size = max(1, n_options // 10)
    segment_counts: dict = Counter(idx // seg_size for idx in pred_indices)

    typed = [r for r in predictions if r["hit_original_option"] is not None]
    n_unparsed = sum(1 for r in predictions if r["parse_failed"])
    n_correct_original = sum(1 for r in typed if r["hit_original_option"] and r["correct"])
    n_wrong_original = sum(1 for r in typed if r["hit_original_option"] and not r["correct"])
    n_distractor = sum(1 for r in typed if not r["hit_original_option"])
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
            f"seg_{s * seg_size}_{s * seg_size + seg_size - 1}": segment_counts.get(s, 0)
            for s in range(10)
        },
        "option_type_stats": option_type_stats,
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 52}")
    print(f"  Model   : {metrics['model']}")
    print(f"  Window  : {metrics['window']}  |  Mode: {metrics['placement_mode']}")
    print(f"  Accuracy: {n_correct}/{len(evaluated)}  ({100 * accuracy:.1f}%)")
    print(f"  Parse failures: {parse_failures}")
    print(f"\n  Per-category accuracy:")
    for cat, v in metrics["per_category"].items():
        print(f"    {cat:<22} {v['correct']:>4}/{v['total']:<4}  ({100 * v['accuracy']:.1f}%)")
    print(f"\n  Per-dependency accuracy:")
    for dep, v in metrics["per_dependency_label"].items():
        print(f"    {dep:<30} {v['correct']:>4}/{v['total']:<4}  ({100 * v['accuracy']:.1f}%)")
    print(f"\n  Per-dependency per-category accuracy:")
    for dep, cat_map in metrics["per_dependency_per_category"].items():
        print(f"    [{dep}]")
        for cat, v in cat_map.items():
            print(f"      {cat:<22} {v['correct']:>4}/{v['total']:<4}  ({100 * v['accuracy']:.1f}%)")
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

    print(f"{'=' * 52}")
    print(f"\nMetrics saved to: {metrics_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    from vllm import LLM, SamplingParams

    try:
        from vllm.sampling_params import GuidedDecodingParams
        _has_guided_params = True
    except ImportError:
        _has_guided_params = False

    # --- Resolve dataset path ---
    if args.baseline:
        dataset_path = "data/bench/mmlu-pro-augmented-10"
        options_key = "options"
        answer_index_key = "answer_index"
        window_str = "baseline"
    else:
        bench_root = f"data/bench/mmlu-pro-augmented-clean-{args.n_options}"
        window_str = f"{args.window:.2f}"
        dataset_path = os.path.join(
            bench_root, f"seed_{args.seed}", f"window_{window_str}", args.mode
        )
        options_key = "new_options"
        answer_index_key = "new_answer_index"

    if not os.path.isdir(dataset_path):
        print(
            f"ERROR: dataset not found at '{dataset_path}'.\n"
            f"Run {'sample_final_dataset.py' if args.baseline else 'augment_final_dataset.py'} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading dataset: {dataset_path}")
    ds = load_from_disk(dataset_path)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    n_options = len(ds[0][options_key])
    if args.baseline:
        print(f"  {len(ds)} questions  |  {n_options} options each  [baseline]")
    else:
        print(f"  {len(ds)} questions  |  {n_options} options each")
        print(f"  seed={args.seed}  window={args.window}  mode={args.mode}")

    indices = [row[answer_index_key] for row in ds]
    print(f"  Gold index — min={min(indices)}  max={max(indices)}  mean={sum(indices)/len(indices):.1f}")

    # --- Output dir ---
    model_slug = args.model_name.split("/")[-1]
    if args.baseline:
        dataset_tag = "mmlu-pro-augmented-10__baseline"
    else:
        dataset_tag = (
            f"mmlu-pro-augmented-{args.n_options}"
            f"__seed_{args.seed}"
            f"__window_{window_str}"
            f"__{args.mode}"
        )
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, model_slug, dataset_tag, now_str)
    os.makedirs(output_dir, exist_ok=True)

    # --- Save run config ---
    config = {
        "model_name": args.model_name,
        "baseline": args.baseline,
        "seed": args.seed if not args.baseline else None,
        "window": args.window if not args.baseline else None,
        "mode": args.mode if not args.baseline else None,
        "n_options": n_options,
        "n_questions": len(ds),
        "limit": args.limit,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "timestamp": now_str,
        "max_tokens": args.max_tokens,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enable_thinking": args.enable_thinking,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nRun config saved to: {output_dir}/run_config.json")

    # --- Sample prompt preview ---
    sample_id, sample_msgs = format_messages(ds[0], n_options, options_key)
    sample_text = sample_msgs[0]["content"]
    print(f"\n--- Sample prompt (question_id={sample_id}) ---")
    sample_lines = sample_text.split("\n")
    preview = sample_lines[:6] + ["  ...", f"  ({n_options - 1}) {ds[0][options_key][-1]}"] + sample_lines[-2:]
    print("\n".join(preview))
    print("--- end sample ---\n")

    # --- Build messages list ---
    custom_ids: List[str] = []
    all_messages: List[list] = []
    for item in ds:
        qid, msgs = format_messages(item, n_options, options_key)
        custom_ids.append(qid)
        all_messages.append(msgs)

    # --- Load model ---
    print(f"Loading model: {args.model_name}")
    llm_kwargs = {"model": args.model_name}
    if args.tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    # --- Sampling params with guided JSON decoding ---
    json_schema = build_json_schema(n_options)
    if _has_guided_params:
        from vllm.sampling_params import GuidedDecodingParams
        guided = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            guided_decoding=guided,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            guided_json=json_schema,
        )

    # --- Run inference ---
    print(f"Running inference on {len(all_messages)} questions...")

    # Qwen3 chat template supports enable_thinking to suppress <think> tokens,
    # which would otherwise interfere with constrained JSON generation.
    is_qwen3 = "qwen3" in args.model_name.lower()
    chat_kwargs: dict = {}
    if is_qwen3:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}

    outputs = llm.chat(all_messages, sampling_params=sampling_params, **chat_kwargs)

    # --- Write raw_completions.jsonl in Together-compatible format ---
    raw_path = os.path.join(output_dir, "raw_completions.jsonl")
    with open(raw_path, "w") as f:
        for qid, output in zip(custom_ids, outputs):
            content = output.outputs[0].text if output.outputs else ""
            record = {
                "custom_id": qid,
                "response": {
                    "body": {
                        "choices": [{"message": {"content": content}}]
                    }
                },
            }
            f.write(json.dumps(record) + "\n")
    print(f"Raw completions saved to: {raw_path}")

    # --- Evaluate ---
    evaluate(raw_path, output_dir, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local vLLM inference on one augmented MMLU-Pro configuration."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.5,
        choices=VALID_WINDOWS,
        metavar="{" + ",".join(str(w) for w in VALID_WINDOWS) + "}",
        help="Sliding window position for the gold answer (0.0=start … 1.0=end).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sparse",
        choices=VALID_MODES,
        help="Placement mode: 'block' or 'sparse'.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model ID (e.g. 'Qwen/Qwen3-27B-FP8', 'meta-llama/Llama-3.3-70B-Instruct').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions",
        help="Base directory for output files.",
    )
    parser.add_argument(
        "--n-options",
        type=int,
        default=1000,
        help="Number of options in the augmented dataset. Used to locate data/bench/mmlu-pro-augmented-clean-<n>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Distractor sampling seed — selects the seed_<N>/ subfolder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run on the original 10-option baseline (data/bench/mmlu-pro-augmented-10). "
             "Ignores --window, --mode, --seed, and --n-options.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate per response (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override vLLM's max_model_len (context window). "
             "Set if default is too small for large option counts.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_false",
        dest="enable_thinking",
        help="Disable Qwen3 thinking mode (enabled by default).",
    )
    parser.set_defaults(enable_thinking=True)

    args = parser.parse_args()
    main(args)
