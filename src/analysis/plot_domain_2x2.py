"""
2×2 figure of stacked horizontal bar charts: strategy distribution (%)
by domain × model, for the four benchmark categories
(computer science, health, law, math).

Two output variants are saved per run:
  • windowed  – y-axis = model groups → window rows  (strategy vs window position)
  • nowindow  – y-axis = model rows only (all windows aggregated)

Usage:
    python src/analysis/plot_domain_2x2.py \\
        --classified-path out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl

    # restrict to one placement mode and/or dependency label:
    python src/analysis/plot_domain_2x2.py \\
        --classified-path ... --mode block --dep option_independent

    # restrict to specific models:
    python src/analysis/plot_domain_2x2.py \\
        --classified-path ... --models "Qwen3.5-27B,GPT-OSS-120B"
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

# 2×2 grid layout: [[top-left, top-right], [bottom-left, bottom-right]]
# Each entry is (category_string, display_label).
DOMAIN_LAYOUT = [
    [("computer science", "Computer Science"), ("health", "Health")],
    [("law",              "Law"),              ("math",  "Math")],
]

TOP_STRATS  = ["S1", "S2", "S3", "S4", "S8"]
OTHER_LABEL = "Other"
ALL_STRATS  = TOP_STRATS + [OTHER_LABEL]

PALETTE = {
    "S1":    "#4C72B0",
    "S2":    "#DD8452",
    "S3":    "#55A868",
    "S4":    "#C44E52",
    "S8":    "#8172B2",
    "Other": "#CCCCCC",
}

MODEL_ABBREV = {
    "Qwen3.5-27B-FP8":        "Qwen3.5-27B",
    "Qwen3.5-9B":             "Qwen3.5-9B",
    "gemini-3-flash-preview":  "Gemini-3-Flash",
    "gemma-4-31B-it":         "Gemma-4-31B",
    "gpt-oss-120b":           "GPT-OSS-120B",
    "gpt-oss-20b":            "GPT-OSS-20B",
}

N_BG_COLORS = ["#EEF3FB", "#FBF3EE", "#EEFBF3"]

DEP_LABELS_ORDERED = ["option_independent", "option_dependent"]
DEP_BG = {
    "option_independent": "#EBF5FB",
    "option_dependent":   "#FEF9E7",
}
DEP_DISPLAY = {
    "option_independent": "Independent",
    "option_dependent":   "Dependent",
}
DEP_LABEL_COLOR = {
    "option_independent": "#1A5276",
    "option_dependent":   "#784212",
}

_CODE_RE   = re.compile(r"^[Ss](\d+)")
_MODEL_GAP = 1.5
_DEP_GAP   = 4.0   # extra space between dep sections within a panel


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_model_slugs(names: list) -> set:
    lower = {n.strip().lower() for n in names}
    result = set()
    for slug, abbrev in MODEL_ABBREV.items():
        if slug.lower() in lower or abbrev.lower() in lower:
            result.add(slug)
    matched_lower = {s.lower() for s in result}
    for n in names:
        if n.strip().lower() not in matched_lower:
            result.add(n.strip())
    return result


def _norm(raw: str) -> str:
    m = _CODE_RE.match(raw.strip())
    return f"S{m.group(1)}" if m else raw


# ── data loading ──────────────────────────────────────────────────────────────

def load_category_map(bench_path: str) -> dict:
    """Build {str(question_id): category} from the benchmark JSONL."""
    cat_map = {}
    with open(bench_path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                cat_map[str(r["question_id"])] = r["category"]
    return cat_map


def load_records(path: str, bench_path: str) -> list:
    cat_map = load_category_map(bench_path)

    records = []
    missing = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                cat = cat_map.get(str(r.get("question_id", "")))
                if cat is None:
                    missing += 1
                    continue
                r["category"] = cat
                records.append(r)

    if missing:
        print(f"  Warning: {missing} records had no matching question_id in bench (skipped)")

    records = [r for r in records if r.get("confidence") in ("high", "medium")]
    for r in records:
        r["primary_strategy"] = _norm(r["primary_strategy"])
    return records


def _pcts_from(recs: list) -> dict:
    total = len(recs)
    if not total:
        return {s: 0.0 for s in ALL_STRATS}
    cnts = Counter(r["primary_strategy"] for r in recs)
    p = {s: 100 * cnts.get(s, 0) / total for s in TOP_STRATS}
    p[OTHER_LABEL] = max(0.0, 100 - sum(p.values()))
    return p


def panel_data_domain(records: list, domain: str,
                       n_filter: Optional[int] = None,
                       model_filter: Optional[set] = None,
                       mode_filter: Optional[str] = None,
                       dep_filter: Optional[str] = None,
                       aggregate_windows: bool = False):
    """
    Returns (n_options_list, rows) for the given domain.

    aggregate_windows=True collapses all window positions into a single row
    per (n_options, model), showing the overall strategy mix.
    """
    subset = [r for r in records if r.get("category") == domain]
    if n_filter is not None:
        subset = [r for r in subset if int(r["n_options"]) == n_filter]
    if model_filter is not None:
        subset = [r for r in subset if r.get("model_slug", "unknown") in model_filter]
    if mode_filter is not None:
        subset = [r for r in subset if r.get("mode") == mode_filter]
    if dep_filter is not None:
        subset = [r for r in subset if r.get("option_dep") == dep_filter]

    if aggregate_windows:
        groups: dict = defaultdict(lambda: defaultdict(list))
        for r in subset:
            groups[int(r["n_options"])][r.get("model_slug", "unknown")].append(r)

        n_options_list = sorted(groups)
        rows = []
        for n_opt in n_options_list:
            for model in sorted(groups[n_opt]):
                rows.append({
                    "n_options": n_opt,
                    "model":     model,
                    "window":    "all",
                    "pcts":      _pcts_from(groups[n_opt][model]),
                })
    else:
        groups2: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in subset:
            groups2[int(r["n_options"])][r.get("model_slug", "unknown")][r.get("window")].append(r)

        n_options_list = sorted(groups2)
        rows = []
        for n_opt in n_options_list:
            for model in sorted(groups2[n_opt]):
                wins = sorted(groups2[n_opt][model],
                              key=lambda v: float(v) if v is not None else 0)
                for w in wins:
                    rows.append({
                        "n_options": n_opt,
                        "model":     model,
                        "window":    w,
                        "pcts":      _pcts_from(groups2[n_opt][model][w]),
                    })

    return n_options_list, rows


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_panel(ax, n_options_list: list, rows: list, title: str,
               show_n_label: bool = True,
               model_fontsize: float = 7.5,
               window_fontsize: float = 7) -> None:
    n = len(rows)
    if n == 0:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        return

    y_pos = np.zeros(n)
    offset = 0.0
    for i in range(n):
        if i > 0:
            prev = rows[i - 1]
            cur  = rows[i]
            if cur["model"] != prev["model"] or cur["n_options"] != prev["n_options"]:
                offset += _MODEL_GAP
        y_pos[i] = i + offset

    lefts = np.zeros(n)
    for strat in ALL_STRATS:
        vals = np.array([r["pcts"].get(strat, 0.0) for r in rows])
        ax.barh(y_pos, vals, left=lefts, color=PALETTE[strat],
                height=0.72, linewidth=0)
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(r["window"]) for r in rows], fontsize=window_fontsize)
    ax.tick_params(axis="y", length=0, pad=1)

    trans = ax.get_yaxis_transform()

    spans: dict = {}
    for i, r in enumerate(rows):
        n_opt = r["n_options"]
        model = r["model"]
        y     = y_pos[i]
        if n_opt not in spans:
            spans[n_opt] = {"y_lo": y, "y_hi": y, "models": {}}
        spans[n_opt]["y_hi"] = y
        mdl = spans[n_opt]["models"]
        if model not in mdl:
            mdl[model] = (y, y)
        else:
            mdl[model] = (mdl[model][0], y)

    for n_idx, n_opt in enumerate(n_options_list):
        sp = spans.get(n_opt)
        if sp is None:
            continue
        n_lo, n_hi = sp["y_lo"], sp["y_hi"]
        n_mid = (n_lo + n_hi) / 2.0

        ax.axhspan(n_lo - 0.5, n_hi + 0.5,
                   color=N_BG_COLORS[n_idx % len(N_BG_COLORS)],
                   zorder=0, alpha=0.45)

        if n_idx > 0:
            ax.axhline(n_lo - 0.5, color="#444444", linewidth=1.4, zorder=2)

        if show_n_label:
            ax.text(1.01, n_mid, f"N={n_opt}", transform=trans,
                    va="center", ha="left", fontsize=6.5, fontweight="bold",
                    color="#333333", clip_on=False)

        models_here = sorted(sp["models"], key=lambda m: sp["models"][m][0])
        for m_idx, model in enumerate(models_here):
            lo, hi = sp["models"][model]
            abbrev = MODEL_ABBREV.get(model, model)

            if m_idx % 2 == 1:
                ax.axhspan(lo - 0.36, hi + 0.36,
                           color="#C8C8C8", zorder=0, alpha=0.25)

            ax.text(50, lo - 0.75, abbrev,
                    transform=ax.transData,
                    va="center", ha="center", fontsize=model_fontsize,
                    color="#111111", fontweight="bold", clip_on=False)

            if m_idx < len(models_here) - 1:
                next_lo = sp["models"][models_here[m_idx + 1]][0]
                sep_y   = (hi + next_lo) / 2.0
                ax.axhline(sep_y, color="#888888", linewidth=0.8,
                           linestyle="--", zorder=1)

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.0, y_pos[-1] + 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Strategy share (%)", fontsize=7.5)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# ── dep-split data + drawing ──────────────────────────────────────────────────

def panel_data_domain_dep_split(records: list, domain: str,
                                  n_filter: Optional[int] = None,
                                  model_filter: Optional[set] = None,
                                  mode_filter: Optional[str] = None,
                                  aggregate_windows: bool = False):
    """
    Returns (n_options_list, rows) where rows are ordered:
        [all independent rows … all dependent rows …]
    Each row carries an extra 'dep_label' key.
    """
    all_rows = []
    for dep in DEP_LABELS_ORDERED:
        _, dep_rows = panel_data_domain(
            records, domain, n_filter, model_filter, mode_filter, dep, aggregate_windows
        )
        for r in dep_rows:
            r["dep_label"] = dep
        all_rows.extend(dep_rows)

    n_options_list = sorted({r["n_options"] for r in all_rows}) if all_rows else []
    return n_options_list, all_rows


def draw_panel_dep_split(ax, rows: list, title: str,
                          model_fontsize: float = 7.5,
                          window_fontsize: float = 7) -> None:
    """Draw a panel with dep_label super-sections → model groups → window rows."""
    n = len(rows)
    if n == 0:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        return

    y_pos = np.zeros(n)
    offset = 0.0
    for i in range(n):
        if i > 0:
            prev = rows[i - 1]
            cur  = rows[i]
            if cur["dep_label"] != prev["dep_label"]:
                offset += _DEP_GAP
            elif cur["model"] != prev["model"]:
                offset += _MODEL_GAP
        y_pos[i] = i + offset

    lefts = np.zeros(n)
    for strat in ALL_STRATS:
        vals = np.array([r["pcts"].get(strat, 0.0) for r in rows])
        ax.barh(y_pos, vals, left=lefts, color=PALETTE[strat],
                height=0.72, linewidth=0)
        lefts += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(r["window"]) for r in rows], fontsize=window_fontsize)
    ax.tick_params(axis="y", length=0, pad=1)

    trans = ax.get_yaxis_transform()

    dep_spans: dict = {}
    model_spans: dict = {}
    models_per_dep: dict = {}

    for i, r in enumerate(rows):
        dep   = r["dep_label"]
        model = r["model"]
        y     = y_pos[i]

        if dep not in dep_spans:
            dep_spans[dep] = {"y_lo": y, "y_hi": y}
            models_per_dep[dep] = []
        dep_spans[dep]["y_hi"] = y

        key = (dep, model)
        if key not in model_spans:
            model_spans[key] = (y, y)
            models_per_dep[dep].append(model)
        else:
            model_spans[key] = (model_spans[key][0], y)

    for d_idx, dep in enumerate(DEP_LABELS_ORDERED):
        sp = dep_spans.get(dep)
        if sp is None:
            continue
        d_lo, d_hi = sp["y_lo"], sp["y_hi"]
        d_mid = (d_lo + d_hi) / 2.0

        ax.axhspan(d_lo - 0.5, d_hi + 0.5, color=DEP_BG[dep], zorder=0, alpha=0.4)

        if d_idx > 0:
            ax.axhline(d_lo - 0.5, color="#333333", linewidth=2.0, zorder=2)

        ax.text(1.02, d_mid, DEP_DISPLAY[dep],
                transform=trans, va="center", ha="left",
                fontsize=7, fontweight="bold",
                color=DEP_LABEL_COLOR[dep], clip_on=False)

        models_here = models_per_dep[dep]
        for m_idx, model in enumerate(models_here):
            lo, hi = model_spans[(dep, model)]
            abbrev = MODEL_ABBREV.get(model, model)

            if m_idx % 2 == 1:
                ax.axhspan(lo - 0.36, hi + 0.36,
                           color="#C8C8C8", zorder=0, alpha=0.25)

            ax.text(50, lo - 0.75, abbrev,
                    transform=ax.transData,
                    va="center", ha="center", fontsize=model_fontsize,
                    color="#111111", fontweight="bold", clip_on=False)

            if m_idx < len(models_here) - 1:
                next_lo = model_spans[(dep, models_here[m_idx + 1])][0]
                sep_y   = (hi + next_lo) / 2.0
                ax.axhline(sep_y, color="#888888", linewidth=0.8,
                           linestyle="--", zorder=1)

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.0, y_pos[-1] + 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Strategy share (%)", fontsize=7.5)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=6)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def build_figure_dep_split(records: list,
                             aggregate_windows: bool = False,
                             n_filter: Optional[int] = None,
                             fig_height: Optional[float] = None,
                             model_filter: Optional[set] = None,
                             mode_filter: Optional[str] = None) -> plt.Figure:
    """2×2 domain figure with independent/dependent super-sections in each panel."""
    cache = {}
    for row_domains in DOMAIN_LAYOUT:
        for domain, _ in row_domains:
            cache[domain] = panel_data_domain_dep_split(
                records, domain, n_filter, model_filter, mode_filter, aggregate_windows
            )

    if fig_height is None:
        max_rows = max(len(rows) for _, rows in cache.values())
        fig_height = max(10.0, max_rows * 0.35 + 4.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, fig_height))
    fig.subplots_adjust(
        left=0.06, right=0.92,
        top=0.94,  bottom=max(0.05, 1.1 / fig_height),
        hspace=0.20, wspace=0.30,
    )

    for ri, row_domains in enumerate(DOMAIN_LAYOUT):
        for ci, (domain, label) in enumerate(row_domains):
            _, rows = cache[domain]
            draw_panel_dep_split(axes[ri][ci], rows, title=label)

    handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in ALL_STRATS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ALL_STRATS),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        title="Strategy",
        title_fontsize=9,
    )

    window_desc = "all windows aggregated" if aggregate_windows else "by window position"
    mode_desc   = f"  ·  mode={mode_filter}" if mode_filter else ""
    fig.suptitle(
        f"Strategy by domain & dependency  ({window_desc}{mode_desc})",
        fontsize=11, fontweight="bold", y=0.975,
    )

    return fig


# ── figure builder ────────────────────────────────────────────────────────────

def build_figure(records: list,
                 aggregate_windows: bool = False,
                 n_filter: Optional[int] = None,
                 fig_height: Optional[float] = None,
                 model_filter: Optional[set] = None,
                 mode_filter: Optional[str] = None,
                 dep_filter: Optional[str] = None) -> plt.Figure:
    """Build the 2×2 domain figure."""
    cache = {}
    for row_domains in DOMAIN_LAYOUT:
        for domain, _ in row_domains:
            cache[domain] = panel_data_domain(
                records, domain, n_filter, model_filter,
                mode_filter, dep_filter, aggregate_windows,
            )

    if fig_height is None:
        max_rows = max(len(rows) for _, rows in cache.values())
        fig_height = max(8.0, max_rows * 0.35 + 2.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, fig_height))
    fig.subplots_adjust(
        left=0.06, right=0.94,
        top=0.94,  bottom=max(0.05, 1.1 / fig_height),
        hspace=0.20, wspace=0.28,
    )

    for ri, row_domains in enumerate(DOMAIN_LAYOUT):
        for ci, (domain, label) in enumerate(row_domains):
            n_options_list, rows = cache[domain]
            draw_panel(axes[ri][ci], n_options_list, rows,
                       title=label,
                       show_n_label=(n_filter is None))

    handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in ALL_STRATS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ALL_STRATS),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        title="Strategy",
        title_fontsize=9,
    )

    window_desc = "all windows aggregated" if aggregate_windows else "by window position"
    mode_desc   = f"  ·  mode={mode_filter}" if mode_filter else ""
    dep_desc    = f"  ·  dep={dep_filter.replace('option_', '')}" if dep_filter else ""
    fig.suptitle(
        f"Strategy distribution by domain  ({window_desc}{mode_desc}{dep_desc})",
        fontsize=11, fontweight="bold", y=0.975,
    )

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    records = load_records(args.classified_path, args.bench_path)
    print(f"Loaded {len(records)} high/medium-confidence records")

    model_filter = None
    if args.models:
        model_filter = resolve_model_slugs([m.strip() for m in args.models.split(",")])
        print(f"Model filter: {sorted(model_filter)}")

    base_out = Path(args.output)
    out_dir  = base_out.parent / "domain2x2"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem   = base_out.stem
    suffix = base_out.suffix

    all_n = sorted({int(r["n_options"]) for r in records if "n_options" in r})

    for aggregate_windows, tag in [(False, "windowed"), (True, "nowindow")]:
        # overall 2×2 (all N values)
        fig = build_figure(records,
                           aggregate_windows=aggregate_windows,
                           fig_height=26,
                           model_filter=model_filter,
                           mode_filter=args.mode,
                           dep_filter=args.dep)
        p = out_dir / f"{stem}_{tag}{suffix}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")

        # per-N 2×2
        for n_val in all_n:
            fig = build_figure(records,
                               aggregate_windows=aggregate_windows,
                               n_filter=n_val,
                               model_filter=model_filter,
                               mode_filter=args.mode,
                               dep_filter=args.dep)
            p = out_dir / f"{stem}_{tag}_N{n_val}{suffix}"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved → {p}")

    # dep-split variant (independent + dependent super-sections within each panel)
    for aggregate_windows, tag in [(False, "windowed"), (True, "nowindow")]:
        fig = build_figure_dep_split(records,
                                     aggregate_windows=aggregate_windows,
                                     fig_height=36,
                                     model_filter=model_filter,
                                     mode_filter=args.mode)
        p = out_dir / f"{stem}_dep_{tag}{suffix}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")

        for n_val in all_n:
            fig = build_figure_dep_split(records,
                                         aggregate_windows=aggregate_windows,
                                         n_filter=n_val,
                                         model_filter=model_filter,
                                         mode_filter=args.mode)
            p = out_dir / f"{stem}_dep_{tag}_N{n_val}{suffix}"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy distribution 2×2 by domain (computer science / health / law / math)."
    )
    parser.add_argument("--classified-path", required=True,
                        help="Path to classified_traces.jsonl")
    parser.add_argument("--bench-path", default="data/bench/mmlu-pro-augmented-10.jsonl",
                        help="Benchmark JSONL with question_id→category mapping "
                             "(default: data/bench/mmlu-pro-augmented-10.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Base output path (default: <classified-dir>/domain_2x2.pdf)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model slugs/abbreviations to include.")
    parser.add_argument("--mode", default=None, choices=["block", "sparse"],
                        help="Restrict to one placement mode (default: both).")
    parser.add_argument("--dep", default=None,
                        choices=["option_independent", "option_dependent"],
                        help="Restrict to one dependency label (default: both).")
    args = parser.parse_args()
    if args.output is None:
        args.output = str(Path(args.classified_path).parent / "domain_2x2.pdf")
    main(args)
