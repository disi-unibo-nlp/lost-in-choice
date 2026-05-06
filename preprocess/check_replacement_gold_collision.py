"""
Sanity check for flagged_distractors.jsonl:

  1. HARD CHECK — no replacement equals the gold answer for that question.
     A gold collision would introduce a second correct answer after
     apply_distractor_fixes.py runs.  Exits with code 1 if any are found.

  2. INFO — duplicate replacement texts per question (two different flagged
     distractors receiving the same replacement).  Duplicates among wrong
     options are acceptable but reported for awareness.

Usage:
    python preprocess/check_replacement_gold_collision.py \\
        --flagged out/distractor_check/.../flagged_distractors.jsonl
"""

import argparse
import json
import statistics
import sys
from collections import Counter


SEP  = "=" * 60
SEP2 = "-" * 60


def main(args):
    records = [json.loads(l) for l in open(args.flagged) if l.strip()]

    total_replacements = sum(r["n_flagged"] for r in records)
    questions_with_flags = [r for r in records if r["n_flagged"] > 0]

    # ── 1. Gold collision check ─────────────────────────────────────────────
    gold_collisions = []
    for r in records:
        gold = r["gold"]
        for item in r.get("flagged", []):
            if item["replacement"] == gold:
                gold_collisions.append({
                    "question_id"          : r["question_id"],
                    "gold"                 : gold,
                    "original_text"        : item["original_text"],
                    "replacement"          : item["replacement"],
                    "global_distractor_idx": item["global_distractor_idx"],
                })

    print(f"\n{SEP}")
    print(f"  REPLACEMENT QUALITY CHECK")
    print(f"  {args.flagged}")
    print(SEP)

    print(f"\n{'Gold-collision check (HARD)'}")
    print(SEP2)
    print(f"  Total replacements checked : {total_replacements:,}")
    if gold_collisions:
        print(f"  FAIL — {len(gold_collisions)} replacement(s) equal the gold answer:\n")
        for c in gold_collisions:
            print(f"    qid={c['question_id']}  global_idx={c['global_distractor_idx']}")
            print(f"      gold        : {c['gold']!r}")
            print(f"      original    : {c['original_text']!r}")
            print(f"      replacement : {c['replacement']!r}")
    else:
        print(f"  OK — no replacement equals the gold answer  ✓")

    # ── 2. Duplicate replacement stats ─────────────────────────────────────
    print(f"\n{'Duplicate replacements (INFO)'}")
    print(SEP2)

    dup_counts   = []   # n_duplicates per question (only questions with flags)
    dup_pcts     = []   # duplicate % per question (only questions with flags)
    dup_details  = []   # (qid, n_flagged, n_dup, pct, examples)

    for r in questions_with_flags:
        replacements = [item["replacement"] for item in r["flagged"]]
        counts = Counter(replacements)
        # a text is duplicate if it appears more than once → excess occurrences
        n_dup = sum(c - 1 for c in counts.values() if c > 1)
        n_total = len(replacements)
        pct = 100 * n_dup / n_total if n_total else 0.0
        dup_counts.append(n_dup)
        dup_pcts.append(pct)
        if n_dup > 0:
            examples = [text for text, c in counts.items() if c > 1]
            dup_details.append((r["question_id"], n_total, n_dup, pct, examples))

    n_with_dups     = sum(1 for d in dup_counts if d > 0)
    total_dup       = sum(dup_counts)
    overall_dup_pct = 100 * total_dup / total_replacements if total_replacements else 0.0

    print(f"  Questions with ≥1 flag         : {len(questions_with_flags)}")
    print(f"  Questions with ≥1 duplicate    : {n_with_dups}  "
          f"({100 * n_with_dups / len(questions_with_flags):.1f}% of flagged questions)"
          if questions_with_flags else "  Questions with ≥1 duplicate    : 0")
    print(f"  Total duplicate replacements   : {total_dup}  "
          f"({overall_dup_pct:.2f}% of all replacements)")

    if dup_pcts:
        nonzero = [p for p in dup_pcts if p > 0]
        if nonzero:
            print(f"\n  Per-question duplicate rate (among questions with ≥1 dup):")
            print(f"    min    : {min(nonzero):.1f}%")
            print(f"    max    : {max(nonzero):.1f}%")
            print(f"    mean   : {statistics.mean(nonzero):.1f}%")
            print(f"    median : {statistics.median(nonzero):.1f}%")

    # ── Per-N impact ────────────────────────────────────────────────────────
    print(f"\n  Duplicate impact relative to total option-list size")
    print(f"  (a duplicate wastes one slot out of N options per question)")
    print(f"  {'N':>6}  {'total_dups':>10}  {'affected_q':>10}  "
          f"{'mean_dup%/q':>12}  {'max_dup%/q':>11}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*11}")
    for n in sorted(args.n_options):
        per_q_pcts = [100 * n_dup / n for _, _, n_dup, _, _ in dup_details if n_dup > 0]
        if per_q_pcts:
            print(f"  {n:>6}  {total_dup:>10}  {len(per_q_pcts):>10}  "
                  f"{statistics.mean(per_q_pcts):>11.3f}%  "
                  f"{max(per_q_pcts):>10.3f}%")
        else:
            print(f"  {n:>6}  {0:>10}  {0:>10}  {'0.000%':>12}  {'0.000%':>11}")

    if dup_details:
        dup_details.sort(key=lambda x: x[2], reverse=True)
        print(f"\n  Questions with most duplicates (top {min(args.top, len(dup_details))}):")
        header = f"  {'qid':<12} {'flagged':>8} {'dups':>6}"
        for n in sorted(args.n_options):
            header += f"  {'dup%@'+str(n):>10}"
        header += "  examples"
        print(header)
        print(f"  {'-'*12} {'-'*8} {'-'*6}" + f"  {'-'*10}" * len(args.n_options) + f"  {'-'*30}")
        for qid, n_total, n_dup, pct, examples in dup_details[:args.top]:
            ex_str = ", ".join(f"{e!r}"[:25] for e in examples[:2])
            if len(examples) > 2:
                ex_str += f" … +{len(examples)-2} more"
            row = f"  {str(qid):<12} {n_total:>8} {n_dup:>6}"
            for n in sorted(args.n_options):
                row += f"  {100*n_dup/n:>9.3f}%"
            row += f"  {ex_str}"
            print(row)

    print(f"\n{SEP}\n")
    sys.exit(1 if gold_collisions else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged", required=True,
                        help="flagged_distractors.jsonl from retrieve_distractor_check.py")
    parser.add_argument("--top", type=int, default=10,
                        help="Questions with most duplicates to show (default: 10)")
    args = parser.parse_args()
    main(args)
