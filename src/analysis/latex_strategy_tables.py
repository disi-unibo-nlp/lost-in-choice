"""
Generate LaTeX tables for strategy distribution across 5 axes:
  dep × N × window × mode × model

Produces 4 tables (one per dep × mode pair) in a single .tex file.
Each table: rows = N × window (15 rows), columns = model.
Each cell shows S1 / S2 / S3 / S4 / S8 proportions (%) stacked via \\makecell.

Required LaTeX packages: booktabs, multirow, makecell

Usage:
    python src/analysis/latex_strategy_tables.py \\
        --classified-path out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl \\
        --out-path out/tables/strategy_tables.tex
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAIN_STRATEGIES = ["S1", "S2", "S3", "S4", "S8"]

# Display order and short names for models
MODEL_ORDER = [
    "gpt-oss-20b",
    "gpt-oss-120b",
    "gemini-3-flash-preview",
    "gemma-4-31B-it",
    "Qwen3.5-9B",
    "Qwen3.5-27B-FP8",
]
MODEL_SHORT = {
    "gpt-oss-20b":            r"GPT-20B",
    "gpt-oss-120b":           r"GPT-120B",
    "gemini-3-flash-preview": r"Gemini-Flash",
    "gemma-4-31B-it":         r"Gemma-31B",
    "Qwen3.5-9B":             r"Qwen-9B",
    "Qwen3.5-27B-FP8":        r"Qwen-27B",
}

DEP_ORDER  = ["option_independent", "option_dependent"]
DEP_LABEL  = {
    "option_independent": "Option-independent",
    "option_dependent":   "Option-dependent",
}
MODE_ORDER = ["block", "sparse"]

_CODE_RE = re.compile(r"^[Ss](\d+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(code: str) -> str:
    m = _CODE_RE.match(code.strip())
    return f"S{m.group(1)}" if m else code.strip()


def _build_dist(records: list) -> dict:
    """Return {strategy: pct} for a list of records."""
    total = len(records)
    if not total:
        return {}
    counts: dict = defaultdict(int)
    for r in records:
        counts[r["primary_strategy"]] += 1
    return {s: round(100 * counts[s] / total, 1) for s in MAIN_STRATEGIES if counts[s]}


def _cell(dist: dict, threshold: float = 1.0) -> str:
    """LaTeX \\makecell with one line per strategy above threshold."""
    lines = [
        f"\\textbf{{{s}}}:{dist[s]:.0f}\\%" if dist.get(s, 0) >= 20
        else f"{s}:{dist[s]:.0f}\\%"
        for s in MAIN_STRATEGIES
        if dist.get(s, 0) >= threshold
    ]
    if not lines:
        return "—"
    return r"\makecell[l]{" + r" \\ ".join(lines) + "}"


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("confidence", "low") not in ("high", "medium"):
                continue
            r["primary_strategy"] = _normalize(r.get("primary_strategy", ""))
            records.append(r)
    print(f"Loaded {len(records)} records (high+medium confidence)")
    return records


def build_table(
    records: list,
    dep: str,
    mode: str,
    models: list,
    n_opts: list,
    windows: list,
) -> str:
    """
    Build a single LaTeX table for one (dep, mode) combination.

    Rows : model (outer) × N (middle) × window (inner)
    Cols : S1 | S2 | S3 | S4 | S8  (percentages)
    """
    dep_short = "indep" if "independent" in dep else "dep"
    label     = f"tab:strat_{dep_short}_{mode}"
    caption   = (
        f"Strategy proportions (\\%) — {DEP_LABEL[dep]}, {mode} mode. "
        r"\textbf{Bold} $\geq$20\%; values $<$1\% omitted."
    )

    strats   = MAIN_STRATEGIES          # S1 S2 S3 S4 S8
    col_spec = "ccc|" + "r" * len(strats)   # Model, N, Window | strategy cols

    lines = []
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"Model & $N$ & Window & "
        + " & ".join(f"\\textbf{{{s}}}" for s in strats)
        + r" \\",
        r"\midrule",
    ]

    rows_per_model = len(n_opts) * len(windows)   # 15

    for m_i, model in enumerate(models):
        rows_per_n = len(windows)                 # 5

        for n_i, n_opt in enumerate(n_opts):
            for w_i, window in enumerate(windows):
                subset = [
                    r for r in records
                    if r.get("model_slug") == model
                    and r.get("option_dep") == dep
                    and str(r.get("n_options")) == n_opt
                    and str(r.get("window")) == window
                    and r.get("mode") == mode
                ]
                dist = _build_dist(subset)

                # Strategy pct cells (right-aligned; bold ≥ 20%)
                s_cells = []
                for s in strats:
                    p = dist.get(s, 0.0)
                    if p < 1.0:
                        s_cells.append("—")
                    elif p >= 20.0:
                        s_cells.append(f"\\textbf{{{p:.0f}}}")
                    else:
                        s_cells.append(f"{p:.0f}")

                # Multirow spans
                model_col = (
                    f"\\multirow{{{rows_per_model}}}{{*}}{{{MODEL_SHORT[model]}}}"
                    if n_i == 0 and w_i == 0 else ""
                )
                n_col = (
                    f"\\multirow{{{rows_per_n}}}{{*}}{{{n_opt}}}"
                    if w_i == 0 else ""
                )

                row = model_col + " & " + n_col + f" & {window} & " \
                      + " & ".join(s_cells) + r" \\"
                lines.append(row)

            # Thin rule between N groups within a model
            if n_i < len(n_opts) - 1:
                lines.append(r"\cmidrule{2-8}")

        # Thicker rule between models (not after the last one)
        if m_i < len(models) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    classified_path = Path(args.classified_path)
    out_path        = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(classified_path)

    # Derive axis values present in the data
    models  = [m for m in MODEL_ORDER if any(r.get("model_slug") == m for r in records)]
    n_opts  = sorted({str(r["n_options"]) for r in records}, key=lambda x: int(x))
    windows = sorted({str(r["window"]) for r in records}, key=float)

    print(f"Models  : {models}")
    print(f"N opts  : {n_opts}")
    print(f"Windows : {windows}")
    print(f"Axis sizes: {len(DEP_ORDER)} dep × {len(n_opts)} N × {len(windows)} window "
          f"× {len(MODE_ORDER)} mode × {len(models)} model = "
          f"{len(DEP_ORDER)*len(n_opts)*len(windows)*len(MODE_ORDER)*len(models)} cells")

    preamble = "\n".join([
        "% Auto-generated by latex_strategy_tables.py",
        "% Required packages: \\usepackage{booktabs,multirow,makecell}",
        "% --------------------------------------------------------",
        "",
    ])

    tables = []
    for dep in DEP_ORDER:
        for mode in MODE_ORDER:
            print(f"  Building table: {dep} × {mode}")
            tables.append(build_table(records, dep, mode, models, n_opts, windows))

    out_path.write_text(preamble + "\n\n".join(tables))
    print(f"\nTables written → {out_path}")
    print(f"Include in LaTeX with: \\input{{{out_path}}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX strategy distribution tables (5-axis)."
    )
    parser.add_argument(
        "--classified-path",
        default="out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl",
        help="Path to classified_traces.jsonl.",
    )
    parser.add_argument(
        "--out-path",
        default="out/tables/strategy_tables.tex",
        help="Output .tex file (default: out/tables/strategy_tables.tex).",
    )
    args = parser.parse_args()
    main(args)
