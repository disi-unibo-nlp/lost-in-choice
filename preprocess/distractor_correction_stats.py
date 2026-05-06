"""
Print and export statistics about flagged distractor corrections.

Metrics computed per domain and overall:
  - n_questions          : total questions in that domain
  - n_flagged_questions  : questions with ≥1 flagged distractor
  - pct_flagged_questions: n_flagged_questions / n_questions * 100
  - avg_flagged_per_q    : mean n_flagged over ALL questions (incl. 0-flag ones)
  - avg_flagged_per_flagged_q : mean n_flagged over flagged questions only
  - avg_dup_replacement_pct   : mean over flagged questions of
                                (n_flagged - n_unique_replacements) / n_flagged * 100
                                i.e. how often the judge recycled the same replacement
                                text for multiple distractors in the same question

Usage:
    python preprocess/distractor_correction_stats.py \\
        --flagged out/distractor_check/merged/block/flagged_distractors.jsonl
"""

import argparse
import json
from pathlib import Path

CATEGORY_ORDER = ["computer science", "math", "law", "health"]
CAT_LABELS = {
    "computer science": "Computer Science",
    "math"            : "Mathematics",
    "law"             : "Law",
    "health"          : "Health",
}


def compute_stats(records: list) -> dict:
    n_q                 = len(records)
    flagged_recs        = [r for r in records if r["n_flagged"] > 0]
    n_flagged_q         = len(flagged_recs)
    total_flagged       = sum(r["n_flagged"] for r in records)

    dup_pcts = []
    for r in flagged_recs:
        reps       = [fi["replacement"] for fi in r["flagged"]]
        n_unique   = len(set(reps))
        dup_pcts.append((len(reps) - n_unique) / len(reps) * 100)

    return {
        "n_questions"              : n_q,
        "n_flagged_questions"      : n_flagged_q,
        "pct_flagged_questions"    : n_flagged_q / n_q * 100 if n_q else 0,
        "total_flagged"            : total_flagged,
        "avg_flagged_per_q"        : total_flagged / n_q if n_q else 0,
        "avg_flagged_per_flagged_q": total_flagged / n_flagged_q if n_flagged_q else 0,
        "avg_dup_replacement_pct"  : sum(dup_pcts) / len(dup_pcts) if dup_pcts else 0,
    }


def fmt_row(label: str, s: dict) -> str:
    return (
        f"  {label:<22s}"
        f"  {s['n_questions']:>5d}"
        f"  {s['n_flagged_questions']:>6d}  ({s['pct_flagged_questions']:5.1f}%)"
        f"  {s['avg_flagged_per_q']:>6.2f}"
        f"  {s['avg_flagged_per_flagged_q']:>7.2f}"
        f"  {s['avg_dup_replacement_pct']:>7.2f}%"
    )


def print_stats(stats_by_cat: dict, overall: dict):
    header = (
        f"  {'Domain':<22s}"
        f"  {'N_q':>5s}"
        f"  {'Flagged_q':>12s}"
        f"  {'Avg_flag/q':>10s}"
        f"  {'Avg_flag/fq':>11s}"
        f"  {'Dup_rep%':>9s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print("\nDistractor correction statistics")
    print("=" * len(header))
    print(header)
    print(sep)
    for cat in CATEGORY_ORDER:
        label = CAT_LABELS.get(cat, cat)
        print(fmt_row(label, stats_by_cat[cat]))
    print(sep)
    print(fmt_row("OVERALL", overall))
    print()
    print("  N_q            : total questions in domain")
    print("  Flagged_q      : questions with ≥1 flagged distractor  (% of N_q)")
    print("  Avg_flag/q     : mean flagged distractors per question  (all questions)")
    print("  Avg_flag/fq    : mean flagged distractors per question  (flagged questions only)")
    print("  Dup_rep%       : mean % of replacement suggestions that are duplicates")
    print("                   within the same question (averaged over flagged questions)")


def build_tex(stats_by_cat: dict, overall: dict) -> str:
    rows = []
    for cat in CATEGORY_ORDER:
        s     = stats_by_cat[cat]
        label = CAT_LABELS.get(cat, cat)
        rows.append(
            f"  {label} & "
            f"{s['n_flagged_questions']} / {s['n_questions']} "
            f"({s['pct_flagged_questions']:.1f}\\%) & "
            f"{s['avg_flagged_per_q']:.2f} & "
            f"{s['avg_flagged_per_flagged_q']:.2f} & "
            f"{s['avg_dup_replacement_pct']:.1f}\\% \\\\"
        )
    s = overall
    rows.append(
        f"  \\midrule\n"
        f"  \\textbf{{Overall}} & "
        f"\\textbf{{{s['n_flagged_questions']} / {s['n_questions']} "
        f"({s['pct_flagged_questions']:.1f}\\%)}} & "
        f"\\textbf{{{s['avg_flagged_per_q']:.2f}}} & "
        f"\\textbf{{{s['avg_flagged_per_flagged_q']:.2f}}} & "
        f"\\textbf{{{s['avg_dup_replacement_pct']:.1f}\\%}} \\\\"
    )

    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Distractor correction statistics per domain. "
        "\\emph{Flagged\\_q}: questions with at least one flagged distractor. "
        "\\emph{Avg/q}: mean flagged distractors per question (all questions). "
        "\\emph{Avg/fq}: mean flagged distractors per flagged question only. "
        "\\emph{Dup\\%}: mean percentage of replacement suggestions that are "
        "duplicate within the same question.}\n"
        "\\label{tab:correction-stats}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Domain & Flagged questions & Avg/q & Avg/fq & Dup\\% \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main(args):
    with open(args.flagged) as f:
        records = [json.loads(l) for l in f if l.strip()]

    stats_by_cat = {}
    for cat in CATEGORY_ORDER:
        cat_recs         = [r for r in records if r.get("category", "").lower() == cat]
        stats_by_cat[cat] = compute_stats(cat_recs)

    overall = compute_stats(records)

    print_stats(stats_by_cat, overall)

    tex = build_tex(stats_by_cat, overall)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"\nLaTeX table written → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged", default="out/distractor_check/merged/block/flagged_distractors.jsonl")
    parser.add_argument("--output",  default="out/distractor_check/merged/correction_stats.tex")
    args = parser.parse_args()
    main(args)
