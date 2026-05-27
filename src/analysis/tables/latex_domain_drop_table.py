"""
Generate two LaTeX tables matching ideal_table_style.tex using real metrics.

Table 1 (compact): model × {Ind, Dep}, columns = 10-opt | Block/Sparse Drop at N=100/500/1000
Table 2 (domain): same but with per-domain (CS, Math, Health, Law, AVG) breakdown

Drops = baseline - min(window_accuracy) — worst-case absolute drop (pp).

Usage:
    python src/analysis/latex_domain_drop_table.py
    python src/analysis/latex_domain_drop_table.py --out-path out/tables/domain_drop.tex
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

BASE = Path("out/completions")
SEED = 42
N_OPTS = [100, 500, 1000]
MODES = ["block", "sparse"]
CATS_SHORT = {
    "computer science": "CS",
    "health": "Health",
    "math": "Math",
    "law": "Law",
}
CAT_ORDER = ["computer science", "math", "health", "law"]

# Number of questions per (dep_type, category) — used for weighted AVG
DEP_CAT_COUNTS = {
    "option_independent": {"computer science": 204, "math": 172, "health": 107, "law": 57},
    "option_dependent":   {"computer science":  46, "math":  78, "health": 143, "law": 193},
}

MODEL_ORDER = [
    "gemini-3-flash-preview",
    "Qwen3.5-27B-FP8",
    "Qwen3.5-9B",
    "gemma-4-31B-it",
    "gpt-oss-120b",
    "gpt-oss-20b",
]
MODEL_DISPLAY = {
    "gemini-3-flash-preview": r"Gemini-3-Flash",
    "Qwen3.5-27B-FP8":        r"Qwen3.5-27B",
    "Qwen3.5-9B":             r"Qwen3.5-9B",
    "gemma-4-31B-it":         r"Gemma-4-31B",
    "gpt-oss-120b":           r"GPT-OSS-120B",
    "gpt-oss-20b":            r"GPT-OSS-20B",
}
DEP_TYPES = ["option_independent", "option_dependent"]
DEP_SHORT  = {"option_independent": "Ind", "option_dependent": "Dep"}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _latest_metrics(exp_dir: Path) -> Optional[dict]:
    ts_dirs = sorted(d for d in exp_dir.iterdir() if d.is_dir())
    if not ts_dirs:
        return None
    mp = ts_dirs[-1] / "metrics.json"
    return json.load(open(mp)) if mp.exists() else None


def _reconstruct_pdpc_from_preds(baseline_dir: Path) -> dict:
    """Build per_dependency_per_category counts from predictions.jsonl."""
    ts_dirs = sorted(d for d in baseline_dir.iterdir() if d.is_dir())
    preds_path = ts_dirs[-1] / "predictions.jsonl"
    counts: dict = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for line in preds_path.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        dep = p.get("option_dependent_vs_independent", "")
        cat = p.get("category", "")
        if not dep or not cat:
            continue
        counts[dep][cat]["total"] += 1
        if p.get("correct"):
            counts[dep][cat]["correct"] += 1
    result = {}
    for dep, cats in counts.items():
        result[dep] = {}
        for cat, v in cats.items():
            acc = v["correct"] / v["total"] if v["total"] else 0.0
            result[dep][cat] = {"correct": v["correct"], "total": v["total"], "accuracy": acc}
    return result


def load_baseline(model_dir: Path) -> Tuple[dict, dict]:
    """
    Returns (per_dep_label, per_dep_per_cat) dicts from baseline metrics.
    per_dep_label: {dep_type: accuracy_float}
    per_dep_per_cat: {dep_type: {cat: accuracy_float}}
    """
    baseline_dir = model_dir / "mmlu-pro-augmented-10__baseline"
    ts_dirs = sorted(d for d in baseline_dir.iterdir() if d.is_dir())
    metrics = json.load(open(ts_dirs[-1] / "metrics.json"))

    per_dep_label = {
        dep: v["accuracy"]
        for dep, v in metrics.get("per_dependency_label", {}).items()
    }

    raw_pdpc = metrics.get("per_dependency_per_category")
    if raw_pdpc is None:
        raw_pdpc = _reconstruct_pdpc_from_preds(baseline_dir)

    per_dep_per_cat = {
        dep: {cat: v["accuracy"] for cat, v in cats.items()}
        for dep, cats in raw_pdpc.items()
    }
    return per_dep_label, per_dep_per_cat


def load_window_metrics(model_dir: Path, n_opts: int, mode: str, seed: int) -> list:
    """Return list of (window, metrics_dict) sorted by window."""
    pattern = re.compile(
        rf"mmlu-pro-augmented-{n_opts}__seed_{seed}__window_(?P<w>[\d.]+)__{mode}"
    )
    results = []
    for exp_dir in model_dir.iterdir():
        m = pattern.fullmatch(exp_dir.name)
        if not m:
            continue
        data = _latest_metrics(exp_dir)
        if data is None:
            continue
        results.append((float(m.group("w")), data))
    return sorted(results, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Drop computation
# ---------------------------------------------------------------------------

def compute_drops(model_dir: Path) -> dict:
    """
    Returns nested dict:
      drops[n][mode] = {
          dep_type: {"overall": float},   # per-dep-label drop (for table 1)
          "per_cat": {cat: float},        # overall per-category drop (for table 2)
      }
    Drops are worst-case: baseline - min(window_accuracy).
    """
    per_dep_label, _ = load_baseline(model_dir)

    # Load overall per-category baseline from metrics.json
    baseline_dir = model_dir / "mmlu-pro-augmented-10__baseline"
    ts_dirs = sorted(d for d in baseline_dir.iterdir() if d.is_dir())
    baseline_metrics = json.load(open(ts_dirs[-1] / "metrics.json"))
    baseline_per_cat = {
        cat: v["accuracy"]
        for cat, v in baseline_metrics.get("per_category", {}).items()
    }

    drops: dict = {}

    for n in N_OPTS:
        drops[n] = {}
        for mode in MODES:
            window_rows = load_window_metrics(model_dir, n, mode, SEED)
            if not window_rows:
                print(f"  [warn] no data: {model_dir.name} n={n} mode={mode}")
                drops[n][mode] = None
                continue

            entry: dict = {}

            # Per-dep-label overall drops (table 1)
            for dep in DEP_TYPES:
                baseline_overall = per_dep_label.get(dep)
                min_overall = min(
                    data.get("per_dependency_label", {}).get(dep, {}).get("accuracy", 1.0)
                    for _, data in window_rows
                )
                if baseline_overall is not None:
                    entry[dep] = {"overall": round(max(0.0, (baseline_overall - min_overall) * 100), 1)}

            # Overall per-category drops (table 2) — no dep split
            per_cat_drops: dict = {}
            for cat in CAT_ORDER:
                baseline_cat = baseline_per_cat.get(cat)
                if baseline_cat is None:
                    continue
                cat_accs = [
                    data.get("per_category", {}).get(cat, {}).get("accuracy", 1.0)
                    for _, data in window_rows
                ]
                min_cat = min(cat_accs)
                per_cat_drops[cat] = round(max(0.0, (baseline_cat - min_cat) * 100), 1)
            entry["per_cat"] = per_cat_drops

            drops[n][mode] = entry

    return drops


# ---------------------------------------------------------------------------
# Table 1: compact (overall dep/ind only)
# ---------------------------------------------------------------------------

def build_table1(all_drops: dict, all_baselines: dict) -> str:
    """Compact table: 10-opt | Block drops N=100/500/1000 | Sparse drops N=100/500/1000."""
    header = (
        r"\begin{table}[!t]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\setlength{\tabcolsep}{3.5pt}" "\n"
        r"\adjustbox{width=\columnwidth}{" "\n"
        r"\begin{tabular}{llccccccc}" "\n"
        r"\toprule" "\n"
        r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Task}} & "
        r"\multirow{2}{*}{\textbf{10-opt}} & "
        r"\multicolumn{3}{c}{\textbf{Block Drop}} & "
        r"\multicolumn{3}{c}{\textbf{Sparse Drop}} \\" "\n"
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}" "\n"
        r" & & & 100 & 500 & 1000 & 100 & 500 & 1000 \\" "\n"
        r"\midrule" "\n"
    )

    rows = []
    for i, model in enumerate(MODEL_ORDER):
        if model not in all_drops:
            continue
        drops   = all_drops[model]
        baselines_dep = all_baselines[model]  # {dep: baseline_acc}
        display = MODEL_DISPLAY[model]

        model_rows = []
        for dep in DEP_TYPES:
            label = DEP_SHORT[dep]
            bl    = baselines_dep.get(dep)
            bl_str = f"{bl*100:.1f}" if bl is not None else "--"

            cells = []
            for mode in MODES:
                for n in N_OPTS:
                    entry = drops.get(n, {}).get(mode)
                    if entry is None or dep not in entry:
                        cells.append("--")
                    else:
                        v = entry[dep].get("overall")
                        cells.append(f"{max(0.0, v):.1f}" if v is not None else "--")

            model_rows.append(f" & {label} & {bl_str} & " + " & ".join(cells) + r" \\")

        # multirow for model name
        model_rows[0] = f"\\multirow{{2}}{{*}}{{{display}}}" + model_rows[0]
        model_rows[1] = "                                " + model_rows[1]
        rows.extend(model_rows)
        if i < len(MODEL_ORDER) - 1:
            rows.append(r"\midrule")

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}" "\n"
        r"\caption{\textbf{Granular performance degradation curves ($\Delta$ percentage points) "
        r"across option scales.} Deltas represent absolute worst-case drops from the 10-option "
        r"baseline for each layout (baseline $-$ min window accuracy).}" "\n"
        r"\label{tab:granular_max_drops}" "\n"
        r"\end{table}" "\n"
    )
    return header + "\n".join(rows) + "\n" + footer


# ---------------------------------------------------------------------------
# Table 2: domain-level breakdown (wide)
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "--"
    return f"{max(0.0, v):.1f}"


def build_table2(all_drops: dict, all_baselines: dict) -> str:
    """Wide table: overall per-domain drops (CS, Math, Health, Law) × N × mode.
    One row per model — no Ind/Dep split, no AVG column.
    Drops use overall per_category accuracy (not per_dep_per_cat).
    """
    # Layout: model | [Block N=100: 4 cols] | sp | [Block N=500] | sp | [Block N=1000] | sp |
    #                [Sparse N=100] | sp | [Sparse N=500] | sp | [Sparse N=1000]
    # 6 groups × 4 domain cols + 5 spacers = 29 data cols, + 1 model col = 30 total
    col_spec = r"l cccc c cccc c cccc c cccc c cccc c cccc"

    def _cat_hdr():
        return " & ".join(r"\textbf{" + CATS_SHORT[c] + r"}" for c in CAT_ORDER)

    header = (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\setlength{\tabcolsep}{2pt}" "\n"
        r"\renewcommand{\arraystretch}{1.2}" "\n"
        r"\adjustbox{width=\linewidth}{" "\n"
        r"\begin{tabular}{" + col_spec + r"}" "\n"
        r"\toprule" "\n"
        r"\multirow{3}{*}{\textbf{Model}} & "
        r"\multicolumn{14}{c}{\textbf{Block}} & & \multicolumn{14}{c}{\textbf{Sparse}} \\" "\n"
        r"\cmidrule(lr){2-15} \cmidrule(lr){17-30}" "\n"
        r" & \multicolumn{4}{c}{\textbf{$N = 100$}} & & "
        r"\multicolumn{4}{c}{\textbf{$N = 500$}} & & "
        r"\multicolumn{4}{c}{\textbf{$N = 1000$}} & & "
        r"\multicolumn{4}{c}{\textbf{$N = 100$}} & & "
        r"\multicolumn{4}{c}{\textbf{$N = 500$}} & & "
        r"\multicolumn{4}{c}{\textbf{$N = 1000$}} \\" "\n"
        r"\cmidrule(lr){2-5} \cmidrule(lr){7-10} \cmidrule(lr){12-15} "
        r"\cmidrule(lr){17-20} \cmidrule(lr){22-25} \cmidrule(lr){27-30}" "\n"
        " & " + " & & ".join([_cat_hdr()] * 6) + r" \\" + "\n"
        r"\midrule" "\n"
    )

    rows = []
    for i, model in enumerate(MODEL_ORDER):
        if model not in all_drops:
            continue
        drops   = all_drops[model]
        display = MODEL_DISPLAY[model]

        cells = []
        for mode in MODES:
            for n in N_OPTS:
                entry = drops.get(n, {}).get(mode)
                per_cat = entry.get("per_cat", {}) if entry else {}
                domain_vals = [_fmt(per_cat.get(cat)) for cat in CAT_ORDER]
                cells.append(" & ".join(domain_vals))

        row = f"{display} & " + " & & ".join(cells) + r" \\"
        rows.append(row)
        if i < len(MODEL_ORDER) - 1:
            rows.append(r"\midrule")

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}" "\n"
        r"\caption{\textbf{Per-domain performance degradation ($\Delta$ absolute pp) "
        r"across option layouts and scales.} "
        r"The layout contrasts \textbf{Block} (left) with \textbf{Sparse} (right). "
        r"Drops are computed as $\text{Baseline} - \min_{\text{window}}$ "
        r"using the overall per-domain accuracy (not split by dependency type).}" "\n"
        r"\label{tab:comprehensive_scaling_appendix_side_by_side}" "\n"
        r"\end{table*}" "\n"
    )
    return header + "\n".join(rows) + "\n" + footer


# ---------------------------------------------------------------------------
# Table 3: domain-first grouping (transpose of Table 2)
# Columns: Block [ CS(100,500,1000) | Math(100,500,1000) | Health | Law ]
#          Sparse [ same ]
# Rows: one per model — easy to compare N-scaling within each domain
# ---------------------------------------------------------------------------

def build_table3(all_drops: dict, all_baselines: dict) -> str:
    """Wide table: same drops as Table 2 but grouped by domain first, then N.
    Layout: Model | Block[CS: 100,500,1000 | Math | Health | Law] | Sparse[same]
    """
    # Per mode: 4 domains × 3 N cols + 3 spacers between domains = 15 cols
    # Full: 1 model + 15 block + 1 spacer + 15 sparse = 32 cols
    col_spec = r"l ccc c ccc c ccc c ccc c ccc c ccc c ccc c ccc"

    def _n_hdr():
        return r"\textbf{100} & \textbf{500} & \textbf{1000}"

    # Column indices (1-based):
    # 1=Model
    # Block: CS=2-4, sp=5, Math=6-8, sp=9, Health=10-12, sp=13, Law=14-16
    # sp=17
    # Sparse: CS=18-20, sp=21, Math=22-24, sp=25, Health=26-28, sp=29, Law=30-32
    header = (
        r"\begin{table*}[t]" "\n"
        r"\centering" "\n"
        r"\small" "\n"
        r"\setlength{\tabcolsep}{2pt}" "\n"
        r"\renewcommand{\arraystretch}{1.2}" "\n"
        r"\adjustbox{width=\linewidth}{" "\n"
        r"\begin{tabular}{" + col_spec + r"}" "\n"
        r"\toprule" "\n"
        r"\multirow{3}{*}{\textbf{Model}} & "
        r"\multicolumn{15}{c}{\textbf{Block}} & & \multicolumn{15}{c}{\textbf{Sparse}} \\" "\n"
        r"\cmidrule(lr){2-16} \cmidrule(lr){18-32}" "\n"
    )

    # Row 2: domain subheaders
    def _domain_spans(offset):
        # offset is the starting column for CS within this mode block
        cs_s, cs_e = offset, offset + 2
        ma_s, ma_e = offset + 4, offset + 6
        he_s, he_e = offset + 8, offset + 10
        la_s, la_e = offset + 12, offset + 14
        cmidrules = (
            f"\\cmidrule(lr){{{cs_s}-{cs_e}}} "
            f"\\cmidrule(lr){{{ma_s}-{ma_e}}} "
            f"\\cmidrule(lr){{{he_s}-{he_e}}} "
            f"\\cmidrule(lr){{{la_s}-{la_e}}}"
        )
        spans = (
            r"\multicolumn{3}{c}{\textbf{CS}} & & "
            r"\multicolumn{3}{c}{\textbf{Math}} & & "
            r"\multicolumn{3}{c}{\textbf{Health}} & & "
            r"\multicolumn{3}{c}{\textbf{Law}}"
        )
        return spans, cmidrules

    block_spans, block_cmidrule = _domain_spans(2)
    sparse_spans, sparse_cmidrule = _domain_spans(18)

    header += " & " + block_spans + " & & " + sparse_spans + r" \\" + "\n"
    header += block_cmidrule + " " + sparse_cmidrule + "\n"

    # Row 3: N subheaders (100 | 500 | 1000 repeated 8 times)
    n_hdr_block  = " & & ".join([_n_hdr()] * 4)
    n_hdr_sparse = " & & ".join([_n_hdr()] * 4)
    header += " & " + n_hdr_block + " & & " + n_hdr_sparse + r" \\" + "\n"
    header += r"\midrule" + "\n"

    rows = []
    for i, model in enumerate(MODEL_ORDER):
        if model not in all_drops:
            continue
        drops   = all_drops[model]
        display = MODEL_DISPLAY[model]

        cells = []
        for mode in MODES:
            domain_groups = []
            for cat in CAT_ORDER:
                n_vals = []
                for n in N_OPTS:
                    entry = drops.get(n, {}).get(mode)
                    per_cat = entry.get("per_cat", {}) if entry else {}
                    n_vals.append(_fmt(per_cat.get(cat)))
                domain_groups.append(" & ".join(n_vals))
            cells.append(" & & ".join(domain_groups))

        row = display + " & " + " & & ".join(cells) + r" \\"
        rows.append(row)
        if i < len(MODEL_ORDER) - 1:
            rows.append(r"\midrule")

    footer = (
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"}" "\n"
        r"\caption{\textbf{Per-domain performance degradation ($\Delta$ absolute pp) "
        r"across option scales, grouped by domain.} "
        r"Same drops as Table~\ref{tab:comprehensive_scaling_appendix_side_by_side} "
        r"re-organised so that scaling curves ($N \in \{100,500,1000\}$) are "
        r"directly comparable within each domain. "
        r"Left half: Block placement; right half: Sparse placement.}" "\n"
        r"\label{tab:domain_first_scaling}" "\n"
        r"\end{table*}" "\n"
    )
    return header + "\n".join(rows) + "\n" + footer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_drops:     dict = {}
    all_baselines: dict = {}

    for model in MODEL_ORDER:
        model_dir = BASE / model
        if not model_dir.exists():
            print(f"[skip] {model} — directory not found")
            continue
        print(f"Processing {model}...")
        per_dep_label, _ = load_baseline(model_dir)
        all_baselines[model] = per_dep_label
        all_drops[model]    = compute_drops(model_dir)

    t1 = build_table1(all_drops, all_baselines)
    t2 = build_table2(all_drops, all_baselines)
    t3 = build_table3(all_drops, all_baselines)

    out_path.write_text(
        "% Auto-generated by latex_domain_drop_table.py\n"
        "% Required packages: booktabs, multirow, adjustbox\n\n"
        + t1 + "\n\n" + t2 + "\n\n" + t3
    )
    print(f"\nWritten → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-path",
        default="out/tables/domain_drop.tex",
        help="Output .tex file path",
    )
    main(parser.parse_args())
