"""
Sanity-check for augment_final_dataset_v2.py output.

Constraints verified for every question in every variant:

  (A) len(new_options) == n_options
  (B) new_options[new_answer_index] == options[answer_index]   (correct answer at declared index)
  (C) new_answer_index == block_start_index + answer_index     (formula: gold = b_start + gold_in_block)
  (D) block mode  : new_options[block_start_index : block_start_index + 10] == options[:10]
                    (original options form a contiguous block, in their original order)
  (E) sparse mode : each original option appears exactly once in new_options
  (F) sparse mode : no two original options share the same segment (seg_size = n // 10)
  (G) CRITICAL: for the same question, block and sparse have the SAME new_answer_index

Usage:
  python3 sanity_check_v2.py --output-root data/bench/mmlu-pro-augmented-100
  python3 sanity_check_v2.py --output-root data/bench/mmlu-pro-augmented-1000 \\
      --seeds 42 --windows 0.00 0.50 1.00 --n-options 1000 --print-samples 3
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import List, Optional

from datasets import load_from_disk

N_ORIG = 10
WINDOW_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def window_label(w: float) -> str:
    return f"window_{w:.2f}"


def load_variant(seed_dir: str, window: float, mode: str):
    path = os.path.join(seed_dir, window_label(window), mode)
    if not os.path.isdir(path):
        return None
    return load_from_disk(path)


def check_question(
    ex_block,
    ex_sparse,
    n: int,
    window: float,
    errors: list,
    qid: str,
):
    """Run all constraint checks for one (block, sparse) pair."""
    seg_size = n // N_ORIG

    orig_opts = list(ex_block["options"])
    ans_idx = ex_block["answer_index"]
    correct_text = orig_opts[ans_idx]
    b_start = ex_block["block_start_index"]

    for mode, ex in [("block", ex_block), ("sparse", ex_sparse)]:
        new_opts = ex["new_options"]
        new_idx = ex["new_answer_index"]
        ex_b_start = ex["block_start_index"]

        # (A) length
        if len(new_opts) != n:
            errors.append(f"[{mode}] qid={qid}: len={len(new_opts)} != {n}")

        # (B) correct text at declared index
        if new_opts[new_idx] != correct_text:
            errors.append(
                f"[{mode}] qid={qid}: new_opts[{new_idx}]={new_opts[new_idx]!r} "
                f"!= correct={correct_text!r}"
            )

        # (C) formula: gold == b_start + answer_index
        expected_gold = ex_b_start + ans_idx
        if new_idx != expected_gold:
            errors.append(
                f"[{mode}] qid={qid}: new_answer_index={new_idx} != "
                f"block_start({ex_b_start}) + answer_index({ans_idx}) = {expected_gold}"
            )

    # (D) block contiguity
    block_opts = ex_block["new_options"]
    block_start_val = ex_block["block_start_index"]
    actual_block_slice = block_opts[block_start_val: block_start_val + N_ORIG]
    if actual_block_slice != orig_opts:
        errors.append(
            f"[block] qid={qid}: slice at [{block_start_val}:{block_start_val+N_ORIG}] "
            f"!= original options"
        )

    # (E) sparse: each original appears exactly once
    sparse_opts = ex_sparse["new_options"]
    orig_set = set(orig_opts)
    for orig in orig_opts:
        count = sparse_opts.count(orig)
        if count != 1:
            errors.append(
                f"[sparse] qid={qid}: original option {orig!r} appears {count} times"
            )

    # (F) sparse: original options in distinct segments
    orig_segs = [sparse_opts.index(o) // seg_size for o in orig_opts]
    if len(set(orig_segs)) != N_ORIG:
        errors.append(
            f"[sparse] qid={qid}: original options share segments — "
            f"segments={orig_segs}"
        )

    # (G) CRITICAL: block and sparse have identical gold index
    block_gold = ex_block["new_answer_index"]
    sparse_gold = ex_sparse["new_answer_index"]
    if block_gold != sparse_gold:
        errors.append(
            f"[MISMATCH] qid={qid}: block gold={block_gold} != sparse gold={sparse_gold}"
        )


def print_sample(ex_block, ex_sparse, n: int, window: float, sample_n: int):
    seg_size = n // N_ORIG
    orig_opts = list(ex_block["options"])
    ans_idx = ex_block["answer_index"]
    correct_text = orig_opts[ans_idx]
    b_start = ex_block["block_start_index"]
    block_gold = ex_block["new_answer_index"]
    sparse_gold = ex_sparse["new_answer_index"]

    q_text = ex_block.get("question", "")
    if len(q_text) > 100:
        q_text = q_text[:97] + "..."

    print(f"\n  --- Sample {sample_n} ---")
    print(f"  question_id   : {ex_block.get('question_id', 'N/A')}")
    print(f"  category      : {ex_block.get('category', 'N/A')}")
    print(f"  question      : {q_text}")
    print(f"  correct text  : {correct_text!r}  (original index {ans_idx})")
    print(f"  block_start   : {b_start}")
    print(f"  block gold idx: {block_gold}  (== {b_start} + {ans_idx})")
    print(f"  sparse gold idx: {sparse_gold}  {'✓ matches block' if block_gold == sparse_gold else '✗ MISMATCH'}")

    # Show block region
    block_opts = ex_block["new_options"]
    print(f"  block region  : new_opts[{b_start}:{b_start+N_ORIG}]")
    for i, opt in enumerate(block_opts[b_start: b_start + N_ORIG]):
        marker = " ← GOLD" if (b_start + i) == block_gold else ""
        print(f"    [{b_start+i:4d}] {opt[:80]!r}{marker}")

    # Show where sparse originals land
    sparse_opts = ex_sparse["new_options"]
    print(f"  sparse originals (one per segment of size {seg_size}):")
    for orig in orig_opts:
        idx = sparse_opts.index(orig)
        seg = idx // seg_size
        marker = " ← GOLD" if idx == sparse_gold else ""
        print(f"    seg {seg:2d}  [{idx:4d}] {orig[:60]!r}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    seeds = args.seeds
    windows = args.windows
    n = args.n_options

    total_questions = 0
    total_errors = 0

    for seed in seeds:
        seed_dir = os.path.join(args.output_root, f"seed_{seed}")
        if not os.path.isdir(seed_dir):
            print(f"[WARN] seed dir not found: {seed_dir}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        for window in windows:
            ds_block = load_variant(seed_dir, window, "block")
            ds_sparse = load_variant(seed_dir, window, "sparse")

            if ds_block is None or ds_sparse is None:
                print(f"  [SKIP] window={window:.2f} — dataset not found")
                continue

            if len(ds_block) != len(ds_sparse):
                print(f"  [ERROR] window={window:.2f}: "
                      f"block has {len(ds_block)} rows, sparse has {len(ds_sparse)} rows")
                total_errors += 1
                continue

            errors: List[str] = []
            samples_printed = 0

            for i, (ex_b, ex_s) in enumerate(zip(ds_block, ds_sparse)):
                qid = str(ex_b.get("question_id", i))
                check_question(ex_b, ex_s, n, window, errors, qid)

                if args.print_samples and samples_printed < args.print_samples:
                    if samples_printed == 0:
                        print(f"\n  [window={window:.2f}] Sample questions:")
                    print_sample(ex_b, ex_s, n, window, samples_printed + 1)
                    samples_printed += 1

            n_q = len(ds_block)
            total_questions += n_q

            if errors:
                total_errors += len(errors)
                print(f"\n  [window={window:.2f}]  {n_q} questions  —  "
                      f"{len(errors)} ERRORS:")
                for e in errors[:20]:
                    print(f"    ✗ {e}")
                if len(errors) > 20:
                    print(f"    ... and {len(errors) - 20} more")
            else:
                b_start_val = ds_block[0]["block_start_index"]
                gold_indices = [r["new_answer_index"] for r in ds_block]
                print(f"  [window={window:.2f}]  {n_q} questions  ✓ all constraints passed  "
                      f"block_start={b_start_val}  "
                      f"gold [{min(gold_indices)}, {max(gold_indices)}]  "
                      f"mean={sum(gold_indices)/len(gold_indices):.1f}")

    print(f"\n{'='*60}")
    if total_errors == 0:
        print(f"  ALL CHECKS PASSED  ({total_questions} questions across all variants)")
    else:
        print(f"  FAILED: {total_errors} errors across {total_questions} questions")
    print(f"{'='*60}\n")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check augment_final_dataset_v2.py output."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/bench/mmlu-pro-augmented-1000",
        help="Root directory of the augmented dataset (same as --output-root in augment_final_dataset_v2.py).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Seeds to check.",
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=WINDOW_POSITIONS,
        help="Window positions to check.",
    )
    parser.add_argument(
        "--n-options",
        type=int,
        default=1000,
        help="Number of options per question.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=2,
        metavar="N",
        help="Print N sample questions per window position (default 2; 0 to disable).",
    )
    args = parser.parse_args()
    sys.exit(main(args))
