"""
Print statistics about the distractor-check intervention.

Reads flagged_distractors.jsonl produced by retrieve_distractor_check.py
(and optionally merged with corrections via merge_distractor_corrections.py).

Usage:
    python preprocess/analysis/distractor_intervention_stats.py \\
        --flagged out/distractor_check/.../flagged_distractors.jsonl
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(num, den):
    return 100 * num / den if den else 0.0


def bucket(n):
    if n == 0:   return "0"
    if n <= 5:   return "1–5"
    if n <= 10:  return "6–10"
    if n <= 20:  return "11–20"
    if n <= 50:  return "21–50"
    return "51+"


BUCKET_ORDER = ["0", "1–5", "6–10", "11–20", "21–50", "51+"]

SEP  = "=" * 60
SEP2 = "-" * 60


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    records = []
    with open(args.flagged) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_questions   = len(records)
    total_checked = sum(r["n_distractors_checked"] for r in records)
    total_flagged = sum(r["n_flagged"] for r in records)
    n_with_flags  = sum(1 for r in records if r["n_flagged"] > 0)

    print(f"\n{SEP}")
    print(f"  DISTRACTOR INTERVENTION STATS")
    print(f"  {args.flagged}")
    print(SEP)

    # ── 1. Overall ──────────────────────────────────────────────────────────
    print(f"\n{'Overall':}")
    print(f"  Questions              : {n_questions}")
    print(f"  Distractors checked    : {total_checked:,}")
    print(f"  Flagged (replaced)     : {total_flagged:,}  ({pct(total_flagged, total_checked):.2f}% of checked)")
    print(f"  Questions with ≥1 flag : {n_with_flags}  ({pct(n_with_flags, n_questions):.1f}% of questions)")

    # Coverage completeness
    incomplete = [r for r in records if r["n_distractors_checked"] < total_checked / n_questions * 0.99]
    if incomplete:
        print(f"  [!] {len(incomplete)} questions with incomplete coverage (<990 checked)")
    else:
        print(f"  Coverage               : complete ({total_checked // n_questions} distractors/question)")

    # ── 2. By category ──────────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  By category")
    print(f"{SEP2}")
    cat_stats = defaultdict(lambda: {"n": 0, "checked": 0, "flagged": 0, "with_flags": 0})
    for r in records:
        c = r.get("category", "unknown")
        cat_stats[c]["n"]          += 1
        cat_stats[c]["checked"]    += r["n_distractors_checked"]
        cat_stats[c]["flagged"]    += r["n_flagged"]
        cat_stats[c]["with_flags"] += int(r["n_flagged"] > 0)

    # Sort by flagged rate descending
    sorted_cats = sorted(cat_stats.items(), key=lambda x: pct(x[1]["flagged"], x[1]["checked"]), reverse=True)
    print(f"  {'category':<22} {'q':>5} {'checked':>8} {'flagged':>8} {'flag%':>7} {'q_with_flag':>12}")
    print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*12}")
    for cat, s in sorted_cats:
        print(f"  {cat:<22} {s['n']:>5} {s['checked']:>8,} {s['flagged']:>8,} "
              f"{pct(s['flagged'], s['checked']):>6.2f}% {s['with_flags']:>12}")

    # ── 3. By option dependency ─────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  By option dependency")
    print(f"{SEP2}")
    dep_stats = defaultdict(lambda: {"n": 0, "checked": 0, "flagged": 0, "with_flags": 0})
    for r in records:
        d = r.get("opt_dep", "unknown")
        dep_stats[d]["n"]          += 1
        dep_stats[d]["checked"]    += r["n_distractors_checked"]
        dep_stats[d]["flagged"]    += r["n_flagged"]
        dep_stats[d]["with_flags"] += int(r["n_flagged"] > 0)

    print(f"  {'opt_dep':<30} {'q':>5} {'checked':>8} {'flagged':>8} {'flag%':>7} {'q_with_flag':>12}")
    print(f"  {'-'*30} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*12}")
    for dep, s in sorted(dep_stats.items()):
        print(f"  {dep:<30} {s['n']:>5} {s['checked']:>8,} {s['flagged']:>8,} "
              f"{pct(s['flagged'], s['checked']):>6.2f}% {s['with_flags']:>12}")

    # ── 4. Flag-count distribution per question ─────────────────────────────
    print(f"\n{SEP2}")
    print(f"  Flags per question (distribution)")
    print(f"{SEP2}")
    flag_counts = [r["n_flagged"] for r in records]
    buckets = Counter(bucket(n) for n in flag_counts)
    for b in BUCKET_ORDER:
        count = buckets.get(b, 0)
        bar   = "█" * (count * 40 // n_questions)
        print(f"  {b:>5} flags : {count:>5} questions  {bar}")

    flagged_counts = [n for n in flag_counts if n > 0]
    if flagged_counts:
        print(f"\n  Among questions with ≥1 flag ({len(flagged_counts)}):")
        print(f"    min    : {min(flagged_counts)}")
        print(f"    max    : {max(flagged_counts)}")
        print(f"    mean   : {statistics.mean(flagged_counts):.1f}")
        print(f"    median : {statistics.median(flagged_counts):.1f}")

    # ── 5. Top flagged questions ─────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  Top {args.top} most-flagged questions")
    print(f"{SEP2}")
    top = sorted(records, key=lambda r: r["n_flagged"], reverse=True)[:args.top]
    for r in top:
        if r["n_flagged"] == 0:
            break
        q_preview = r["question"][:70] + ("…" if len(r["question"]) > 70 else "")
        print(f"  qid={r['question_id']}  flagged={r['n_flagged']}  cat={r['category']}")
        print(f"    Q: {q_preview}")
        print(f"    gold: {r['gold']!r}")
        # Show first 3 replacements as examples
        for item in r["flagged"][:3]:
            orig = item["original_text"][:55]
            repl = item["replacement"][:55]
            print(f"    [{item['global_distractor_idx']:>4}] {orig!r}")
            print(f"           → {repl!r}")
        if r["n_flagged"] > 3:
            print(f"         ... and {r['n_flagged'] - 3} more")
        print()

    # ── 6. Replacement collision summary ────────────────────────────────────
    print(f"{SEP2}")
    print(f"  Replacement quality checks")
    print(f"{SEP2}")
    gold_collisions = 0
    orig_collisions = 0
    for r in records:
        gold = r["gold"]
        orig_set = set()  # we don't have orig options here, skip orig check
        for item in r["flagged"]:
            if item["replacement"] == gold:
                gold_collisions += 1
    if gold_collisions:
        print(f"  [FAIL] Replacements equal to gold answer : {gold_collisions}")
    else:
        print(f"  Replacements equal to gold answer : 0  ✓")
    print(f"  (run check_replacement_gold_collision.py for the full check)")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print stats about flagged_distractors.jsonl interventions."
    )
    parser.add_argument("--flagged", required=True,
                        help="flagged_distractors.jsonl from retrieve_distractor_check.py")
    parser.add_argument("--top", type=int, default=10,
                        help="Number of top-flagged questions to show (default: 10)")
    args = parser.parse_args()
    main(args)
