"""
Merge fails_corrected.json into flagged_distractors.jsonl.

fails_corrected.json is a list of records keyed by batch key
({question_id}__{batch_idx:03d}) with compact flagged items:
    [{"key": "8127__001", "flagged": [{"id": <option_idx_in_batch>, "reason": ..., "replacement": ...}]}]

The script reconstructs the full flagged item structure (original_text,
global_distractor_idx) from the augmented dataset, then merges into
flagged_distractors.jsonl updating n_distractors_checked / n_flagged.
Also rewrites summary.json with updated aggregate counts.

Usage:
    python preprocess/merge_distractor_corrections.py \\
        --output-dir out/distractor_check/gemini-3-flash-preview/block/math/2026-04-25_01-44-10 \\
        --data       data/bench/mmlu-pro-augmented-1000/seed_42/window_0.00/block.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

BATCH_SIZE = 100


def load_dataset(data_path: Path) -> dict:
    """Load augmented dataset indexed by question_id."""
    records = {}
    with open(data_path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                records[str(d["question_id"])] = d
    return records


def get_distractors(ex: dict) -> list:
    orig_set = set(ex["options"])
    return [o for o in ex["new_options"] if o not in orig_set]


def main(args):
    output_dir = Path(args.output_dir)
    corrected_path = output_dir / "fails_corrected.json"
    flagged_path   = output_dir / "flagged_distractors.jsonl"
    summary_path   = output_dir / "summary.json"

    corrections = json.loads(corrected_path.read_text())
    print(f"Corrections loaded: {len(corrections)} batch records")

    # Group corrections by question_id
    by_qid: dict = defaultdict(list)  # qid -> list of (b_idx, flagged_items)
    for entry in corrections:
        key = entry["key"]
        qid, b_idx_str = key.rsplit("__", 1)
        by_qid[qid].append((int(b_idx_str), entry["flagged"]))

    print(f"Loading dataset from: {args.data}")
    dataset = load_dataset(Path(args.data))
    print(f"  {len(dataset)} questions loaded")

    records = []
    with open(flagged_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Existing flagged_distractors.jsonl: {len(records)} records")

    merged_q = 0
    for rec in records:
        qid = str(rec["question_id"])
        if qid not in by_qid:
            continue

        ex = dataset.get(qid)
        if ex is None:
            print(f"  [WARN] qid={qid} not found in dataset — skipping")
            continue

        distractors = get_distractors(ex)

        for b_idx, flagged_items in sorted(by_qid[qid]):
            batch_start = b_idx * BATCH_SIZE
            batch = distractors[batch_start : batch_start + BATCH_SIZE]

            n_added = 0
            for item in flagged_items:
                opt_idx = item["id"]
                if opt_idx >= len(batch):
                    print(f"  [WARN] qid={qid} batch={b_idx}: id={opt_idx} out of range "
                          f"(batch size={len(batch)}) — skipping item")
                    continue
                rec["flagged"].append({
                    "batch_idx"            : b_idx,
                    "option_idx_in_batch"  : opt_idx,
                    "global_distractor_idx": batch_start + opt_idx,
                    "original_text"        : batch[opt_idx],
                    "reason"               : item["reason"],
                    "replacement"          : item["replacement"],
                })
                n_added += 1

            rec["n_distractors_checked"] += len(batch)
            rec["n_flagged"] = len(rec["flagged"])
            print(f"  qid={qid} batch={b_idx}: +{len(batch)} checked, "
                  f"+{n_added} flagged → "
                  f"total checked={rec['n_distractors_checked']} flagged={rec['n_flagged']}")

        # Re-sort flagged items by (batch_idx, option_idx_in_batch) for consistency
        rec["flagged"].sort(key=lambda x: (x["batch_idx"], x["option_idx_in_batch"]))
        merged_q += 1

    with open(flagged_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nUpdated: {flagged_path}")

    # Rewrite summary.json
    total_checked = sum(r["n_distractors_checked"] for r in records)
    total_flagged = sum(r["n_flagged"] for r in records)
    n_with_flags  = sum(1 for r in records if r["n_flagged"] > 0)
    summary = {
        "n_questions"           : len(records),
        "n_questions_with_flags": n_with_flags,
        "n_distractors_checked" : total_checked,
        "n_flagged"             : total_flagged,
        "pct_flagged"           : round(100 * total_flagged / total_checked, 2) if total_checked else 0,
        "parse_failures"        : 0,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Updated: {summary_path}")

    print(f"\nMerged corrections for {merged_q} question(s) "
          f"({len(corrections)} batch records across {len(by_qid)} unique qids)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Directory containing fails_corrected.json and flagged_distractors.jsonl")
    parser.add_argument("--data", required=True,
                        help="Augmented dataset JSONL (same one used for check_distractor_gemini.py)")
    args = parser.parse_args()
    main(args)
