"""
Combine option_dependent_vs_independent predictions from multiple models
using majority voting, with gpt-oss-120b as the primary tiebreaker.

Voting rules:
  - 3 models present: majority wins (2+/3); if all disagree → gpt-oss-120b
  - 2 models present (one missing): if the two agree → use that; else → gpt-oss-120b
  - 1 model present: use gpt-oss-120b if available, else the only available prediction
  - 0 models present: label as "missing"
"""

import json
import os
import argparse
from collections import Counter
from datasets import load_from_disk, Dataset

PRIMARY_MODEL = "gpt-oss-120b"
VALID_LABELS = {"option_dependent", "option_independent"}


def load_predictions(categorized_dir: str) -> dict[str, dict[str, str]]:
    """Load per-model predictions keyed by question_id.

    Returns: {model_name: {question_id: label}}
    """
    preds: dict[str, dict[str, str]] = {}
    for model_name in os.listdir(categorized_dir):
        model_path = os.path.join(categorized_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        ds = load_from_disk(model_path)
        preds[model_name] = {}
        for row in ds:
            qid = str(row["question_id"])
            label = row.get("option_dependent_vs_independent", "missing")
            preds[model_name][qid] = label if label in VALID_LABELS else "missing"
        print(f"Loaded {len(preds[model_name])} predictions from model: {model_name}")
    return preds


def majority_vote(votes: list[str], primary_pred: str | None) -> tuple[str, str]:
    """Return (final_label, method) using majority voting with primary fallback."""
    valid_votes = [v for v in votes if v in VALID_LABELS]

    if not valid_votes:
        return "missing", "no_valid_predictions"

    counts = Counter(valid_votes)
    most_common_label, most_common_count = counts.most_common(1)[0]

    # Majority (strict: more than half of valid votes)
    if most_common_count > len(valid_votes) / 2:
        return most_common_label, "majority"

    # All valid votes tie → fall back to primary model
    if primary_pred and primary_pred in VALID_LABELS:
        return primary_pred, "primary_tiebreak"

    # Primary also missing → pick the most common among valid (arbitrary tiebreak)
    return most_common_label, "arbitrary_tiebreak"


def combine(categorized_dir: str, base_dataset_dir: str, output_path: str):
    preds = load_predictions(categorized_dir)
    models = list(preds.keys())
    print(f"\nModels found: {models}")

    # Collect all question IDs across all models
    all_qids = set()
    for model_preds in preds.values():
        all_qids.update(model_preds.keys())
    print(f"Total unique question IDs: {len(all_qids)}")

    # Load base dataset for question metadata
    print(f"Loading base dataset from: {base_dataset_dir}")
    base_ds = load_from_disk(base_dataset_dir)
    base_rows = {str(row["question_id"]): row for row in base_ds}

    # --- Per-question voting ---
    method_counts: Counter = Counter()
    results = []

    for qid in sorted(all_qids, key=lambda x: int(x) if x.isdigit() else x):
        votes = {m: preds[m].get(qid, "missing") for m in models}
        primary_pred = votes.get(PRIMARY_MODEL)

        final_label, method = majority_vote(list(votes.values()), primary_pred)
        method_counts[method] += 1

        row = dict(base_rows[qid]) if qid in base_rows else {"question_id": qid}
        row["option_dependent_vs_independent"] = final_label
        row["vote_method"] = method
        for m in models:
            row[f"pred_{m}"] = votes[m]
        results.append(row)

    # --- Save ---
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    combined_ds = Dataset.from_list(results)
    combined_ds.save_to_disk(output_path)
    print(f"\nCombined dataset saved (HF format) to: {output_path}")

    jsonl_path = f"{output_path}.jsonl"
    combined_ds.to_json(jsonl_path, orient="records", lines=True)
    print(f"Combined dataset saved (JSONL format) to: {jsonl_path}")

    # --- Statistics ---
    print("\n--- Vote Method Breakdown ---")
    for method, count in method_counts.most_common():
        print(f"  {method}: {count} ({100 * count / len(results):.1f}%)")

    label_counts = Counter(r["option_dependent_vs_independent"] for r in results)
    print("\n--- Final Label Distribution ---")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} ({100 * count / len(results):.1f}%)")

    # Per-model agreement with final label
    print("\n--- Per-Model Agreement with Final Label ---")
    for m in models:
        agreed = sum(
            1 for r in results
            if r[f"pred_{m}"] == r["option_dependent_vs_independent"]
            and r[f"pred_{m}"] in VALID_LABELS
        )
        print(f"  {m}: {agreed}/{len(results)} ({100 * agreed / len(results):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine per-model categorization predictions via majority voting."
    )
    parser.add_argument(
        "--categorized-dir",
        type=str,
        default="mmlu_pro_downsampled_2858_categorized",
        help="Directory containing one subfolder per model with HF datasets.",
    )
    parser.add_argument(
        "--base-dataset-dir",
        type=str,
        default="mmlu_pro_downsampled_2858",
        help="Original downsampled dataset directory (for question metadata).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="mmlu_pro_downsampled_2858_categorized_combined",
        help="Output path for the combined HF dataset.",
    )
    args = parser.parse_args()

    combine(
        categorized_dir=args.categorized_dir,
        base_dataset_dir=args.base_dataset_dir,
        output_path=args.output_path,
    )
