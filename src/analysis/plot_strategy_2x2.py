"""
2×2 figure of stacked horizontal bar charts: strategy distribution (%)
by N-options × model × window, for each (mode × dep_label) combination.

Layout:
  rows    = mode  (block | sparse)
  cols    = dep   (independent | dependent)
  y-axis  = N-options sections → model groups (rotated label) → window rows
  x-axis  = cumulative strategy percentage 0–100 %
  colors  = S1, S2, S3, S4, S8, Other

Usage:
    python src/analysis/plot_strategy_2x2.py \\
        --classified-path out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl

    # custom output path:
    python src/analysis/plot_strategy_2x2.py \\
        --classified-path ... --output figures/strategy_2x2.pdf
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

MODES      = ["block", "sparse"]
DEP_LABELS = ["option_independent", "option_dependent"]

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

PANEL_TITLES = {
    ("block",  "option_independent"): "Block · Independent",
    ("block",  "option_dependent"):   "Block · Dependent",
    ("sparse", "option_independent"): "Sparse · Independent",
    ("sparse", "option_dependent"):   "Sparse · Dependent",
}

MODEL_ABBREV = {
    "Qwen3.5-27B-FP8":        "Qwen3.5-27B",
    "Qwen3.5-9B":             "Qwen3.5-9B",
    "gemini-3-flash-preview":  "Gemini-3-Flash",
    "gemma-4-31B-it":         "Gemma-4-31B",
    "gpt-oss-120b":           "GPT-OSS-120B",
    "gpt-oss-20b":            "GPT-OSS-20B",
}

# section background colors cycling over N-options groups
N_BG_COLORS = ["#EEF3FB", "#FBF3EE", "#EEFBF3"]


def resolve_model_slugs(names: list) -> set:
    """Return the set of data-level model slugs matching any of *names*.

    Each name is matched case-insensitively against both the raw slug and its
    display abbreviation in MODEL_ABBREV.  Unknown names are passed through
    unchanged so callers can use raw slugs directly.
    """
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

_CODE_RE = re.compile(r"^[Ss](\d+)")


# ── data loading ──────────────────────────────────────────────────────────────

def _norm(raw: str) -> str:
    m = _CODE_RE.match(raw.strip())
    return f"S{m.group(1)}" if m else raw


def load_records(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = [r for r in records if r.get("confidence") in ("high", "medium")]
    for r in records:
        r["primary_strategy"] = _norm(r["primary_strategy"])
    return records


def panel_data(records: list, mode: str, dep_label: str,
               n_filter: Optional[int] = None,
               model_filter: Optional[set] = None):
    """
    Returns:
        n_options_list  – sorted list of N values present in this slice
        rows            – list of row dicts {n_options, model, window, pcts}

    model_filter: set of model slugs to include (None = all).
    """
    subset = [
        r for r in records
        if r.get("mode") == mode and r.get("option_dep") == dep_label
    ]
    if n_filter is not None:
        subset = [r for r in subset if int(r["n_options"]) == n_filter]
    if model_filter is not None:
        subset = [r for r in subset if r.get("model_slug", "unknown") in model_filter]

    groups: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in subset:
        groups[int(r["n_options"])][r.get("model_slug", "unknown")][r.get("window")].append(r)

    n_options_list = sorted(groups)
    rows = []
    for n_opt in n_options_list:
        for model in sorted(groups[n_opt]):
            wins = sorted(groups[n_opt][model],
                          key=lambda v: float(v) if v is not None else 0)
            for w in wins:
                recs  = groups[n_opt][model][w]
                total = len(recs)
                cnts  = Counter(r["primary_strategy"] for r in recs)
                pcts  = {s: 100 * cnts.get(s, 0) / total for s in TOP_STRATS} \
                        if total else {s: 0.0 for s in TOP_STRATS}
                pcts[OTHER_LABEL] = max(0.0, 100 - sum(pcts.values()))
                rows.append({
                    "n_options": n_opt,
                    "model":     model,
                    "window":    w,
                    "pcts":      pcts,
                })
    return n_options_list, rows


# ── drawing ───────────────────────────────────────────────────────────────────

_MODEL_GAP = 1.5   # extra vertical space inserted between model groups


def draw_panel(ax, n_options_list: list, rows: list, title: str,
               show_n_label: bool = True,
               model_fontsize: float = 7.5,
               window_fontsize: float = 7) -> None:
    n = len(rows)
    if n == 0:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)
        return

    # y positions with a gap between every model boundary
    y_pos = np.zeros(n)
    offset = 0.0
    for i in range(n):
        if i > 0:
            prev = rows[i - 1]
            cur  = rows[i]
            if cur["model"] != prev["model"] or cur["n_options"] != prev["n_options"]:
                offset += _MODEL_GAP
        y_pos[i] = i + offset

    # stacked horizontal bars
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

    # build span index using actual y coordinates
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

        # sort models by visual (y) order
        models_here = sorted(sp["models"], key=lambda m: sp["models"][m][0])
        for m_idx, model in enumerate(models_here):
            lo, hi = sp["models"][model]
            mid    = (lo + hi) / 2.0
            abbrev = MODEL_ABBREV.get(model, model)

            if m_idx % 2 == 1:
                ax.axhspan(lo - 0.36, hi + 0.36,
                           color="#C8C8C8", zorder=0, alpha=0.25)

            # model name above the bar group (must be < lo-0.36 in data coords
            # since inverted y-axis means smaller y = visually higher)
            ax.text(50, lo - 0.75, abbrev,
                    transform=ax.transData,
                    va="center", ha="center", fontsize=model_fontsize,
                    color="#111111", fontweight="bold", clip_on=False)

            # dashed separator drawn at the midpoint of the gap to the next model
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


# ── figure builder ────────────────────────────────────────────────────────────

def build_figure(records: list, n_filter: Optional[int] = None,
                 fig_height: Optional[float] = None,
                 model_filter: Optional[set] = None) -> plt.Figure:
    """Build the 2×2 strategy figure, optionally restricted to one N value."""
    cache = {}
    for mode in MODES:
        for dep_label in DEP_LABELS:
            cache[(mode, dep_label)] = panel_data(records, mode, dep_label,
                                                   n_filter, model_filter)

    if fig_height is None:
        max_rows = max(len(rows) for _, rows in cache.values())
        fig_height = max(8.0, max_rows * 0.35 + 2.5)

    fig, axes = plt.subplots(2, 2, figsize=(14, fig_height))
    fig.subplots_adjust(
        left=0.06, right=0.94,
        top=0.96,  bottom=max(0.05, 1.1 / fig_height),
        hspace=0.18, wspace=0.28,
    )

    for ri, mode in enumerate(MODES):
        for ci, dep_label in enumerate(DEP_LABELS):
            n_options_list, rows = cache[(mode, dep_label)]
            draw_panel(axes[ri][ci], n_options_list, rows,
                       PANEL_TITLES[(mode, dep_label)],
                       show_n_label=(n_filter is None))

    for ri, mode in enumerate(MODES):
        axes[ri][1].annotate(
            mode.upper(),
            xy=(1.10, 0.5), xycoords="axes fraction",
            va="center", ha="left", fontsize=9,
            fontweight="bold", color="#555555", rotation=-90,
        )

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

    return fig


# ── single-panel builder ──────────────────────────────────────────────────────

def build_single_panel(records: list, mode: str, dep_label: str,
                        n_filter: Optional[int] = None,
                        merge_s8_sparse: bool = True,
                        model_filter: Optional[set] = None) -> Optional[plt.Figure]:
    """1×1 figure for one (mode, dep_label) panel – no title, no mode label.

    merge_s8_sparse: when True (default) and mode=='sparse', S8 is folded into
    Other before rendering, since S8 is only interpretable relative to block.
    Pass False to keep S8 as a separate segment (original behaviour).

    model_filter: set of model slugs to include (None = all).
    """
    n_options_list, rows = panel_data(records, mode, dep_label, n_filter, model_filter)
    if not rows:
        return None

    if mode == "sparse" and merge_s8_sparse:
        for r in rows:
            r["pcts"]["Other"] = r["pcts"].get("Other", 0.0) + r["pcts"].get("S8", 0.0)
            r["pcts"]["S8"] = 0.0

    max_rows = len(rows)
    fig_height = max(5.0, max_rows * 0.28 + 2.0)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, fig_height))
    fig.subplots_adjust(
        left=0.07, right=0.93,
        top=0.97,  bottom=max(0.07, 0.9 / fig_height),
    )

    draw_panel(ax, n_options_list, rows, title="",
               show_n_label=(n_filter is None),
               model_fontsize=13.5,
               window_fontsize=11)
    ax.set_xlabel("")
    return fig


# ── standalone legend ─────────────────────────────────────────────────────────

def build_legend_figure(include_s8: bool = True) -> plt.Figure:
    """Minimal figure containing only the strategy legend (resizable).

    include_s8: when False, S8 is omitted (use for sparse-only legends where
    S8 has been folded into Other).
    """
    strats  = ALL_STRATS if include_s8 else [s for s in ALL_STRATS if s != "S8"]
    handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in strats]
    fig, ax = plt.subplots(figsize=(5.5 if not include_s8 else 6, 0.55))
    ax.set_axis_off()
    fig.legend(
        handles=handles,
        loc="center",
        ncol=len(strats),
        fontsize=11,
        frameon=False,
        title="Strategy",
        title_fontsize=11,
    )
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    records = load_records(args.classified_path)
    print(f"Loaded {len(records)} high/medium-confidence records")

    model_filter = None
    if args.models:
        model_filter = resolve_model_slugs([m.strip() for m in args.models.split(",")])
        print(f"Model filter: {sorted(model_filter)}")

    base_out = Path(args.output)
    out_dir  = base_out.parent / "strategy2x2"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem   = base_out.stem    # "strategy_2x2"
    suffix = base_out.suffix  # ".pdf"

    # standalone legends (full + sparse variant without S8)
    for include_s8, tag in [(True, ""), (False, "_sparse")]:
        leg_path = out_dir / f"{stem}_legend{tag}{suffix}"
        fig = build_legend_figure(include_s8=include_s8)
        fig.savefig(leg_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {leg_path}")

    # overall 2×2
    fig = build_figure(records, fig_height=26, model_filter=model_filter)
    p = out_dir / f"{stem}{suffix}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    # per-N: 2×2 + four 1×1 crops (no panel/mode titles)
    all_n = sorted({int(r["n_options"]) for r in records if "n_options" in r})
    for n_val in all_n:
        # full 2×2
        fig = build_figure(records, n_filter=n_val, model_filter=model_filter)
        n_path = out_dir / f"{stem}_N{n_val}{suffix}"
        fig.savefig(n_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {n_path}")

        # individual 1×1 panels
        for mode in MODES:
            for dep_label in DEP_LABELS:
                dep_slug   = dep_label.replace("option_", "")
                panel_path = out_dir / f"{stem}_N{n_val}_{mode}_{dep_slug}{suffix}"
                fig = build_single_panel(records, mode, dep_label, n_val,
                                         model_filter=model_filter)
                if fig is not None:
                    fig.savefig(panel_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    print(f"Saved → {panel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified-path", required=True,
                        help="Path to classified_traces.jsonl")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <classified-dir>/strategy_2x2.pdf)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated list of models to display "
                             "(slug or abbreviation, e.g. 'Qwen3.5-27B,GPT-OSS-120B,Gemma-4-31B'). "
                             "Default: all models.")
    args = parser.parse_args()
    if args.output is None:
        args.output = str(Path(args.classified_path).parent / "strategy_2x2.pdf")
    main(args)
