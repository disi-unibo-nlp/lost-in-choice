"""
Sanity check for augmented datasets under data/bench/mmlu-pro-augmented-1000.

Two checks per question:
  1. CATEGORY CHECK  — every distractor added to new_options (i.e. options not in
     the original 10) must come from the same-category pool built from the source
     dataset (mmlu_pro_downsampled_2858).
  2. GOLD DUPLICATE CHECK — the correct answer text must appear exactly once in
     new_options (at new_answer_index and nowhere else).

Usage:
    python sanity_check_augmented.py
    python sanity_check_augmented.py \\
        --augmented-root data/bench/mmlu-pro-augmented-1000 \\
        --source-dataset mmlu_pro_downsampled_2858
"""

import argparse
import os
from collections import defaultdict
from datasets import load_from_disk


def build_category_pools(source_ds) -> dict:
    pools = defaultdict(set)
    for ex in source_ds:
        for opt in ex["options"]:
            pools[ex["category"]].add(opt)
    return {cat: texts for cat, texts in pools.items()}


def check_variant(ds, category_pools: dict, label: str) -> dict:
    """
    Check all questions in one dataset variant.
    Returns a summary dict with violation counts and examples.
    """
    category_violations = []   # distractor from wrong category
    gold_duplicate_violations = []  # gold text appears more than once

    for row in ds:
        qid = row["question_id"]
        cat = row["category"]
        original_set = set(row["options"])
        new_options = row["new_options"]
        gold_idx = row["new_answer_index"]
        gold_text = new_options[gold_idx]

        # ---- Check 1: category membership of added distractors ----
        pool = category_pools.get(cat, set())
        for opt in new_options:
            if opt not in original_set and opt not in pool:
                category_violations.append({
                    "question_id": qid,
                    "category": cat,
                    "bad_distractor": opt,
                })
                break  # one violation per question is enough to flag it

        # ---- Check 2: gold text uniqueness in new_options ----
        gold_count = new_options.count(gold_text)
        if gold_count != 1:
            gold_duplicate_violations.append({
                "question_id": qid,
                "category": cat,
                "gold_text": gold_text,
                "occurrences": gold_count,
                "new_answer_index": gold_idx,
            })

    n = len(ds)
    ok = len(category_violations) == 0 and len(gold_duplicate_violations) == 0
    status = "OK" if ok else "FAIL"

    print(f"\n  [{status}] {label}  ({n} questions)")
    if category_violations:
        print(f"    CATEGORY violations : {len(category_violations)}")
        for v in category_violations[:3]:
            print(f"      qid={v['question_id']}  cat={v['category']}  "
                  f"bad_distractor={v['bad_distractor'][:60]!r}")
        if len(category_violations) > 3:
            print(f"      ... and {len(category_violations) - 3} more")
    if gold_duplicate_violations:
        print(f"    GOLD DUPLICATE violations : {len(gold_duplicate_violations)}")
        for v in gold_duplicate_violations[:3]:
            print(f"      qid={v['question_id']}  cat={v['category']}  "
                  f"occurrences={v['occurrences']}  idx={v['new_answer_index']}  "
                  f"gold={v['gold_text'][:50]!r}")
        if len(gold_duplicate_violations) > 3:
            print(f"      ... and {len(gold_duplicate_violations) - 3} more")

    return {
        "label": label,
        "n_questions": n,
        "category_violations": len(category_violations),
        "gold_duplicate_violations": len(gold_duplicate_violations),
        "ok": ok,
    }


def find_variants(augmented_root: str):
    """
    Yield (label, path) for every HF dataset directory found under augmented_root.
    Expected layout: <root>/seed_*/window_*/<mode>/
    """
    for seed_dir in sorted(os.listdir(augmented_root)):
        seed_path = os.path.join(augmented_root, seed_dir)
        if not (os.path.isdir(seed_path) and seed_dir.startswith("seed_")):
            continue
        for window_dir in sorted(os.listdir(seed_path)):
            window_path = os.path.join(seed_path, window_dir)
            if not (os.path.isdir(window_path) and window_dir.startswith("window_")):
                continue
            for mode_dir in sorted(os.listdir(window_path)):
                mode_path = os.path.join(window_path, mode_dir)
                if not os.path.isdir(mode_path):
                    continue
                # An HF dataset directory contains dataset_info.json
                if os.path.isfile(os.path.join(mode_path, "dataset_info.json")):
                    label = f"{seed_dir}/{window_dir}/{mode_dir}"
                    yield label, mode_path


def main(args):
    print(f"Loading source dataset from: {args.source_dataset}")
    source_ds = load_from_disk(args.source_dataset)
    print(f"  {len(source_ds)} questions in source")

    category_pools = build_category_pools(source_ds)
    print(f"  {len(category_pools)} categories: {sorted(category_pools)}")

    variants = list(find_variants(args.augmented_root))
    if not variants:
        print(f"\nNo dataset variants found under: {args.augmented_root}")
        return

    print(f"\nChecking {len(variants)} variants under: {args.augmented_root}")

    results = []
    for label, path in variants:
        ds = load_from_disk(path)
        result = check_variant(ds, category_pools, label)
        results.append(result)

    # ---- Summary ----
    n_ok = sum(1 for r in results if r["ok"])
    n_fail = len(results) - n_ok
    total_cat = sum(r["category_violations"] for r in results)
    total_dup = sum(r["gold_duplicate_violations"] for r in results)

    print(f"\n{'='*60}")
    print(f"  Variants checked  : {len(results)}")
    print(f"  Passed            : {n_ok}")
    print(f"  Failed            : {n_fail}")
    print(f"  Category violations (total) : {total_cat}")
    print(f"  Gold duplicate violations   : {total_dup}")
    print(f"{'='*60}")
    if n_fail == 0:
        print("  All checks passed.")
    else:
        print("  Some checks FAILED — see details above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check augmented MMLU-Pro datasets."
    )
    parser.add_argument(
        "--augmented-root",
        type=str,
        default="data/bench/mmlu-pro-augmented-1000",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default="mmlu_pro_downsampled_2858",
        help="HF dataset used to build per-category distractor pools.",
    )
    args = parser.parse_args()
    main(args)
