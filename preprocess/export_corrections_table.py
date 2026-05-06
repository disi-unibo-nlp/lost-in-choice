"""
Print the first N flagged distractor corrections per domain and export them
as a LaTeX longtable file for the paper appendix.

Usage:
    python preprocess/export_corrections_table.py \\
        --flagged  out/distractor_check/merged/block/flagged_distractors.jsonl \\
        --output   out/distractor_check/merged/corrections_table.tex \\
        --n        10
"""

import argparse
import json
import textwrap
from pathlib import Path

CATEGORY_ORDER = ["computer science", "math", "law", "health"]

CAT_LABELS = {
    "computer science": "Computer Science",
    "math"            : "Mathematics",
    "law"             : "Law",
    "health"          : "Health",
}

# Maximum characters before truncation (None = no truncation)
TRUNC = {
    "question"    : 110,
    "original"    : 80,
    "replacement" : 80,
    "reason"      : 160,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate(text: str, limit: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    # Cut at last space before limit
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "\\ldots{}"


def tex_escape(text: str) -> str:
    """Escape LaTeX special chars that are NOT part of math expressions."""
    # Only escape characters that are unambiguously non-math
    text = text.replace("%", r"\%")
    text = text.replace("&", r"\&")
    text = text.replace("#", r"\#")
    return text


def fmt(text: str, field: str) -> str:
    limit = TRUNC.get(field)
    t = truncate(text, limit) if limit else text.replace("\n", " ").strip()
    return tex_escape(t)


# ---------------------------------------------------------------------------
# Debug print
# ---------------------------------------------------------------------------

def debug_print(by_cat: dict, n: int):
    for cat in CATEGORY_ORDER:
        items = by_cat.get(cat, [])
        print(f"\n{'='*70}")
        print(f"  {CAT_LABELS.get(cat, cat).upper()}  —  {len(items)} total flagged items")
        print(f"{'='*70}")
        for i, (question, gold, fi) in enumerate(items[:n], 1):
            print(f"\n  [{i}] Q: {truncate(question, 100)}")
            print(f"      Gold:        {gold}")
            print(f"      Distractor:  {fi['original_text']}")
            print(f"      Replacement: {fi['replacement']}")
            print(f"      Reason:      {truncate(fi['reason'], 120)}")


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

TEX_HEADER = r"""\documentclass{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{microtype}
\usepackage{amsmath,amssymb}

\begin{document}

\appendix
\section*{Appendix: Distractor Correction Examples}

For each domain we show the first examples of distractors flagged as
potentially correct by our judge, together with the suggested replacement.
\textit{Original} is the distractor as generated; \textit{Replacement} is
the corrected option; \textit{Reason} is the judge's explanation.

"""

TEX_FOOTER = r"""
\end{document}
"""

TABLE_COLS = (
    r">{\raggedright\arraybackslash}X "
    r">{\raggedright\arraybackslash}p{2.5cm} "
    r">{\raggedright\arraybackslash}p{2.2cm} "
    r">{\raggedright\arraybackslash}p{2.2cm} "
    r">{\raggedright\arraybackslash}X"
)
TABLE_HEAD = (
    r"\textbf{Question} & \textbf{Gold} & \textbf{Distractor} & "
    r"\textbf{Replacement} & \textbf{Reason} \\"
)
N_COLS = 5


def build_tex(by_cat: dict, n: int) -> str:
    lines = [TEX_HEADER]

    lines.append(
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Distractor correction examples per domain}\n"
        "\\label{tab:corrections}\n"
        "\\begin{tabularx}{\\textwidth}{" + TABLE_COLS + "}\n"
        "\\toprule\n"
        + TABLE_HEAD + "\n"
    )

    for i, cat in enumerate(CATEGORY_ORDER):
        items = by_cat.get(cat, [])
        label = CAT_LABELS.get(cat, cat)

        lines.append(
            "\\midrule\n"
            f"\\multicolumn{{{N_COLS}}}{{c}}{{\\textit{{{label}}}}}\\\\\n"
            "\\midrule\n"
        )

        for question, gold, fi in items[:n]:
            q_cell  = fmt(question,            "question")
            g_cell  = fmt(gold,                "original")
            o_cell  = fmt(fi["original_text"], "original")
            r_cell  = fmt(fi["replacement"],   "replacement")
            rs_cell = fmt(fi["reason"],        "reason")
            lines.append(f"{q_cell} & {g_cell} & {o_cell} & {r_cell} & {rs_cell} \\\\\n")

    lines.append(
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        "\\end{table}\n"
    )

    lines.append(TEX_FOOTER)
    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    with open(args.flagged) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Collect one (question, gold, flagged_item) triple per distinct question
    # per category — takes the first flagged item from each question that has any.
    by_cat: dict = {cat: [] for cat in CATEGORY_ORDER}
    for rec in records:
        cat = rec.get("category", "").lower()
        if cat not in by_cat or not rec["flagged"]:
            continue
        by_cat[cat].append((rec["question"], rec["gold"], rec["flagged"][0]))

    debug_print(by_cat, args.n)

    tex = build_tex(by_cat, args.n)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"\n\nLaTeX written → {out}")
    for cat in CATEGORY_ORDER:
        print(f"  {CAT_LABELS.get(cat, cat):20s}: {len(by_cat.get(cat, []))} flagged items, "
              f"showing first {args.n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flagged", default="out/distractor_check/merged/block/flagged_distractors.jsonl",
                        help="Merged flagged_distractors.jsonl")
    parser.add_argument("--output",  default="out/distractor_check/merged/corrections_table.tex",
                        help="Output .tex file")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of examples per domain to include")
    args = parser.parse_args()
    main(args)
