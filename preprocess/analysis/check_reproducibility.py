"""
Verify that two independent runs of augment_final_dataset_v2.py produce
bit-for-bit identical outputs for every matching seed / window / mode.

For each seed found in both roots the script checks every question:
  - new_options list is identical (same texts, same order)
  - new_answer_index is identical

Usage:
    python check_reproducibility.py --dir-a data/bench/mmlu-pro-augmented-1000 \\
                                    --dir-b data/bench/old/mmlu-pro-augmented-1000
    python check_reproducibility.py --dir-a prova --dir-b prova-2
"""

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def find_seeds(root: Path):
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("seed_"))


def find_windows(seed_dir: Path):
    return sorted(d for d in seed_dir.iterdir() if d.is_dir() and d.name.startswith("window_"))


def find_modes(window_dir: Path):
    return sorted(
        p.stem for p in window_dir.iterdir()
        if p.suffix == ".jsonl" and not p.stem.startswith(".")
    )


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_variant(rows_a: list, rows_b: list, label: str):
    """Compare two loaded JSONL lists question by question. Returns (n_questions, n_diffs, errors)."""

    errors = []

    if len(rows_a) != len(rows_b):
        errors.append(f"row count: A={len(rows_a)}  B={len(rows_b)}")
        return len(rows_a), len(errors), errors

    n_opt_diff = 0
    n_idx_diff = 0
    first_opt_diff = None
    first_idx_diff = None

    for ra, rb in zip(rows_a, rows_b):
        qid = ra.get("question_id", "?")

        if ra["new_options"] != rb["new_options"]:
            n_opt_diff += 1
            if first_opt_diff is None:
                for j, (a, b) in enumerate(zip(ra["new_options"], rb["new_options"])):
                    if a != b:
                        first_opt_diff = f"qid={qid}  first diff at index {j}: A={a!r}  B={b!r}"
                        break

        if ra["new_answer_index"] != rb["new_answer_index"]:
            n_idx_diff += 1
            if first_idx_diff is None:
                first_idx_diff = (
                    f"qid={qid}  A gold={ra['new_answer_index']}  B gold={rb['new_answer_index']}"
                )

    if n_opt_diff:
        errors.append(f"{n_opt_diff}/{len(rows_a)} questions have different new_options")
        if first_opt_diff:
            errors.append(f"  first: {first_opt_diff}")
    if n_idx_diff:
        errors.append(f"{n_idx_diff}/{len(rows_a)} questions have different new_answer_index")
        if first_idx_diff:
            errors.append(f"  first: {first_idx_diff}")

    return len(rows_a), len(errors), errors


# ---------------------------------------------------------------------------
# Head-to-head sample print
# ---------------------------------------------------------------------------

def print_sample(ra, rb, label: str, ctx: int = 3):
    """Print first question head-to-head between dir-A and dir-B."""
    qid  = ra.get("question_id", "?")
    gold = ra["new_answer_index"]
    lo   = max(0, gold - ctx)
    hi   = min(len(ra["new_options"]), gold + ctx + 1)

    print(f"\n  ┌─ {label}  qid={qid}  gold={gold}")
    print(f"  │  question: {ra.get('question', '')[:100]}")
    print(f"  │")
    print(f"  │  {'idx':>4}   {'Dir-A':^45}   {'Dir-B':^45}   match")
    print(f"  │  {'─'*4}   {'─'*45}   {'─'*45}   {'─'*5}")

    for idx in range(lo, hi):
        oa = ra["new_options"][idx]
        ob = rb["new_options"][idx]
        match = "✓" if oa == ob else "✗"
        marker = " ←GOLD" if idx == gold else "      "
        ta = (oa[:42] + "...") if len(oa) > 45 else oa
        tb = (ob[:42] + "...") if len(ob) > 45 else ob
        print(f"  │  [{idx:>3}]{marker}  {ta:<45}   {tb:<45}   {match}")

    print(f"  └{'─'*110}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    for d in (dir_a, dir_b):
        if not d.is_dir():
            print(f"ERROR: directory not found: {d}", file=sys.stderr)
            sys.exit(1)

    seeds_a = {d.name: d for d in find_seeds(dir_a)}
    seeds_b = {d.name: d for d in find_seeds(dir_b)}

    common = sorted(set(seeds_a) & set(seeds_b))
    only_a = sorted(set(seeds_a) - set(seeds_b))
    only_b = sorted(set(seeds_b) - set(seeds_a))

    print(f"\nDir A : {dir_a}")
    print(f"Dir B : {dir_b}")
    print(f"Seeds in common : {common}")
    if only_a:
        print(f"Seeds only in A : {only_a}")
    if only_b:
        print(f"Seeds only in B : {only_b}")

    total_questions = 0
    total_diffs = 0

    for seed_name in common:
        seed_a = seeds_a[seed_name]
        seed_b = seeds_b[seed_name]

        windows_a = {d.name: d for d in find_windows(seed_a)}
        windows_b = {d.name: d for d in find_windows(seed_b)}
        common_windows = sorted(set(windows_a) & set(windows_b))

        print(f"\n{'='*68}")
        print(f"  {seed_name}")
        print(f"{'='*68}")

        for win_name in common_windows:
            win_a = windows_a[win_name]
            win_b = windows_b[win_name]

            modes_a = set(find_modes(win_a))
            modes_b = set(find_modes(win_b))
            common_modes = sorted(modes_a & modes_b)

            for mode in common_modes:
                jsonl_a = win_a / f"{mode}.jsonl"
                jsonl_b = win_b / f"{mode}.jsonl"

                if not jsonl_a.exists() or not jsonl_b.exists():
                    print(f"  [{win_name}/{mode}]  MISSING jsonl")
                    continue

                rows_a = load_jsonl(jsonl_a)
                rows_b = load_jsonl(jsonl_b)
                n_q, n_err, errors = compare_variant(rows_a, rows_b,
                                                     f"{seed_name}/{win_name}/{mode}")
                total_questions += n_q

                if errors:
                    total_diffs += n_err
                    print(f"  [{win_name}/{mode}]  ✗  {n_q} questions")
                    for e in errors:
                        print(f"      {e}")
                else:
                    print(f"  [{win_name}/{mode}]  ✓  {n_q} questions identical")

                if rows_a and rows_b:
                    print_sample(rows_a[0], rows_b[0],
                                 label=f"{win_name}/{mode}",
                                 ctx=args.ctx)

    print(f"\n{'='*68}")
    if total_diffs == 0:
        print(f"  ALL IDENTICAL — {total_questions} question-variants checked across {len(common)} seeds")
    else:
        print(f"  FAILED — {total_diffs} error groups across {total_questions} question-variants")
    print(f"{'='*68}\n")

    sys.exit(0 if total_diffs == 0 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check two augmentation runs are bit-for-bit identical seed by seed."
    )
    parser.add_argument("--dir-a", required=True,
                        help="Root directory of first run (contains seed_42/, seed_123/, ...)")
    parser.add_argument("--dir-b", required=True,
                        help="Root directory of second run")
    parser.add_argument("--ctx", type=int, default=3,
                        help="Options to show before and after gold in each sample (default 3)")
    args = parser.parse_args()
    main(args)
