"""
For the first N questions in the dataset, print the question text and the
prompt that was sent in the batch file for the same question ID.

Usage:
    python inspect_batch_alignment.py
    python inspect_batch_alignment.py \\
        --dataset-path data/bench/mmlu-pro-augmented-1000/seed_42/window_0.00/sparse \\
        --batch-path  out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__seed_42__window_0.00__sparse/2026-04-23_12-53-12/batch_requests.jsonl \\
        --n 10
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_from_disk

CTX = 5   # distractors to show before and after the block


def print_block_context(row):
    """Print the 5 slots before and after the original-options block."""
    new_opts   = row["new_options"]
    orig_opts  = set(row["options"])
    gold_idx   = row["new_answer_index"]
    n          = len(new_opts)

    # Anchor: use stored block_start_index if present, else min position of any original option
    bs = row.get("block_start_index")
    if bs is None:
        positions = [i for i, o in enumerate(new_opts) if o in orig_opts]
        bs = min(positions) if positions else 0

    lo = max(0, bs - CTX)
    hi = min(n, bs + 10 + CTX)

    print(f"\n  [DATASET — context around block (block_start={bs})]")
    for idx in range(lo, hi):
        opt = new_opts[idx]
        if idx == gold_idx:
            tag = "GOLD  "
        elif opt in orig_opts:
            tag = "orig  "
        else:
            tag = "      "

        # Section dividers
        if idx == bs:
            print(f"  {'─'*66}")
            print(f"    ← original options block starts here")
        if idx == bs + 10:
            print(f"    ← original options block ends here")
            print(f"  {'─'*66}")

        print(f"  [{idx:4d}] {tag} {opt[:70]!r}")


def main(args):
    dataset_path = Path(args.dataset_path)
    batch_path = Path(args.batch_path)

    if not dataset_path.is_dir():
        print(f"ERROR: dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    if not batch_path.is_file():
        print(f"ERROR: batch file not found: {batch_path}", file=sys.stderr)
        sys.exit(1)

    ds = load_from_disk(str(dataset_path))

    with open(batch_path) as f:
        batch_by_qid = {
            str(req["custom_id"]): req
            for line in f if line.strip()
            for req in [json.loads(line)]
        }

    samples = list(ds.select(range(min(args.n, len(ds)))))

    for i, row in enumerate(samples):
        qid = str(row["question_id"])
        sep = "=" * 72

        print(f"\n{sep}")
        print(f"  SAMPLE {i+1}/{len(samples)}   question_id = {qid}")
        print(sep)

        print(f"\n  [DATASET]")
        print(f"  question : {row['question']}")
        print(f"  gold idx : {row['new_answer_index']}  →  {row['new_options'][row['new_answer_index']]!r}")
        print_block_context(row)

        req = batch_by_qid.get(qid)
        if req is None:
            print(f"\n  [BATCH]  ✗ custom_id={qid} not found in batch file")
        else:
            content = req["body"]["messages"][0]["content"]
            print(f"\n  [BATCH PROMPT]")
            print(content)

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="data/bench/mmlu-pro-augmented-1000/seed_42/window_0.00/sparse",
    )
    parser.add_argument(
        "--batch-path",
        default="out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__seed_42__window_0.00__sparse/2026-04-23_12-53-12/batch_requests.jsonl",
    )
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    main(args)
