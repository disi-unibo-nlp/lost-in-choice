"""
2×2 figure of line plots: mean completion_tokens per window position.

completion_tokens is the canonical reasoning-effort metric produced by
analyze_reasoning.py:
  - OpenAI-compat models  → usage.completion_tokens (or n_completion_tokens)
  - Gemini                → thoughtsTokenCount + candidatesTokenCount
  - Qwen3.5-27B (vllm)   → n_completion_tokens

Run analyze_reasoning.py first to produce reasoning_records.jsonl, then:

    python src/analysis/plot_reasoning_2x2.py \\
        --records-path out/reasoning_analysis/reasoning_records.jsonl

    # filter models and/or N:
    python src/analysis/plot_reasoning_2x2.py \\
        --records-path out/reasoning_analysis/reasoning_records.jsonl \\
        --models "Qwen3.5-27B,GPT-OSS-120B,Gemini-3-Flash" \\
        --n-options 1000

Layout:
  rows    = mode  (block | sparse)
  cols    = dep   (independent | dependent)
  x-axis  = window position (0.0 → 1.0)
  y-axis  = mean completion_tokens
  lines   = one per model (color) × N-options (line style when multiple N shown)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

MODES      = ["block", "sparse"]
DEP_LABELS = ["option_independent", "option_dependent"]

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

MODEL_COLORS = {
    "Qwen3.5-27B-FP8":        "#4C72B0",
    "Qwen3.5-9B":             "#DD8452",
    "gemini-3-flash-preview":  "#55A868",
    "gemma-4-31B-it":         "#C44E52",
    "gpt-oss-120b":           "#8172B2",
    "gpt-oss-20b":            "#937860",
}

N_LINESTYLES = {100: ":", 500: "--", 1000: "-"}
N_COLORS     = {100: "#DD8452", 500: "#55A868", 1000: "#4C72B0"}
WINDOWS      = [0.0, 0.25, 0.5, 0.75, 1.0]

COMBO_LINESTYLES = {
    ("block",  "option_independent"): "-",
    ("block",  "option_dependent"):   "--",
    ("sparse", "option_independent"): ":",
    ("sparse", "option_dependent"):   "-.",
}
COMBO_LABELS = {
    ("block",  "option_independent"): "Block · Indep",
    ("block",  "option_dependent"):   "Block · Dep",
    ("sparse", "option_independent"): "Sparse · Indep",
    ("sparse", "option_dependent"):   "Sparse · Dep",
}


# ── data ──────────────────────────────────────────────────────────────────────

def load_records(path: str) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {path}")
    return records


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


def panel_data(records: list, mode: str, dep_label: str,
               n_filter: Optional[int] = None,
               model_filter: Optional[set] = None) -> dict:
    """Return {model_slug: {n_options: {window: mean_completion_tokens}}}."""
    subset = [r for r in records
              if r["mode"] == mode and r["option_dep"] == dep_label
              and r.get("completion_tokens", 0) > 0]
    if n_filter is not None:
        subset = [r for r in subset if r["n_options"] == n_filter]
    if model_filter is not None:
        subset = [r for r in subset if r["model_slug"] in model_filter]

    groups: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in subset:
        groups[r["model_slug"]][r["n_options"]][r["window"]].append(
            r["completion_tokens"])

    result: dict = {}
    for model, n_dict in groups.items():
        result[model] = {}
        for n_opt, w_dict in n_dict.items():
            result[model][n_opt] = {
                w: float(np.mean(v)) for w, v in w_dict.items()
            }
    return result


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_panel(ax, data: dict, title: str, n_list: list,
               show_title: bool = True,
               label_fontsize: float = 8) -> None:
    single_n = len(n_list) == 1

    for model_slug in sorted(data):
        color  = MODEL_COLORS.get(model_slug, "#333333")
        abbrev = MODEL_ABBREV.get(model_slug, model_slug)
        for n_opt in sorted(data[model_slug]):
            if n_opt not in n_list:
                continue
            ls     = N_LINESTYLES.get(n_opt, "-")
            w_dict = data[model_slug][n_opt]
            xs = sorted(w_dict)
            ys = [w_dict[x] for x in xs]
            label = abbrev if single_n else f"{abbrev} (N={n_opt})"
            ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.0,
                    marker="o", markersize=6, label=label, zorder=3)

    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS], fontsize=label_fontsize)
    ax.set_xlabel("Window position", fontsize=label_fontsize)
    ax.set_ylabel("Mean completion tokens", fontsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if show_title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)


# ── figure builders ───────────────────────────────────────────────────────────

def build_figure(records: list, n_filter: Optional[int] = None,
                 model_filter: Optional[set] = None) -> plt.Figure:
    cache: dict = {}
    all_n: set  = set()
    for mode in MODES:
        for dep_label in DEP_LABELS:
            d = panel_data(records, mode, dep_label, n_filter, model_filter)
            cache[(mode, dep_label)] = d
            for m_dict in d.values():
                all_n.update(m_dict.keys())
    n_list = sorted(all_n)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.12,
                        hspace=0.40, wspace=0.32)

    for ri, mode in enumerate(MODES):
        for ci, dep_label in enumerate(DEP_LABELS):
            draw_panel(axes[ri][ci], cache[(mode, dep_label)],
                       PANEL_TITLES[(mode, dep_label)], n_list)

    for ri, mode in enumerate(MODES):
        axes[ri][1].annotate(
            mode.upper(),
            xy=(1.05, 0.5), xycoords="axes fraction",
            va="center", ha="left", fontsize=9,
            fontweight="bold", color="#555555", rotation=-90,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    return fig


def panel_data_for_model(records: list, mode: str, dep_label: str,
                         model_slug: str) -> dict:
    """Return {n_options: {window: mean_completion_tokens}} for one model."""
    subset = [r for r in records
              if r["mode"] == mode and r["option_dep"] == dep_label
              and r["model_slug"] == model_slug
              and r.get("completion_tokens", 0) > 0]
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in subset:
        groups[r["n_options"]][r["window"]].append(r["completion_tokens"])
    return {
        n_opt: {w: float(np.mean(v)) for w, v in w_dict.items()}
        for n_opt, w_dict in groups.items()
    }


def draw_panel_model(ax, data: dict, title: str,
                     show_title: bool = True,
                     show_ylabel: bool = True,
                     show_xlabel: bool = True,
                     label_fontsize: float = 5.5) -> None:
    """Draw one panel for a single model — one line per N value, colored by N."""
    for n_opt in sorted(data):
        color  = N_COLORS.get(n_opt, "#333333")
        ls     = N_LINESTYLES.get(n_opt, "-")
        w_dict = data[n_opt]
        xs = sorted(w_dict)
        ys = [w_dict[x] for x in xs]
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.0,
                marker="o", markersize=2.5, label=f"N={n_opt}", zorder=3)

    ax.set_xticks(WINDOWS)
    if show_xlabel:
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=label_fontsize - 0.5)
        ax.set_xlabel("Window", fontsize=label_fontsize, labelpad=2)
    else:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel("Mean tokens", fontsize=label_fontsize, labelpad=2)
    else:
        ax.set_ylabel("")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda val, _: f"{val / 1000:.2g}K" if val >= 1000 else str(int(val))
        )
    )
    ax.tick_params(axis="y", labelsize=label_fontsize - 0.5, labelleft=show_ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#DDDDDD", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    if show_title:
        ax.set_title(title, fontsize=label_fontsize + 0.5, fontweight="bold", pad=3)


def build_model_figure(records: list, model_slug: str) -> plt.Figure:
    """2×2 figure for a single model — lines = N values (100 / 500 / 1000)."""
    abbrev = MODEL_ABBREV.get(model_slug, model_slug)
    FS = 5.5

    # collect all panel data first to compute shared y range
    panel_cache: dict = {}
    all_vals: list = []
    for mode in MODES:
        for dep_label in DEP_LABELS:
            data = panel_data_for_model(records, mode, dep_label, model_slug)
            panel_cache[(mode, dep_label)] = data
            for n_dict in data.values():
                all_vals.extend(n_dict.values())

    if all_vals:
        pad = (max(all_vals) - min(all_vals)) * 0.08
        y_lo = max(0, min(all_vals) - pad)
        y_hi = max(all_vals) + pad
    else:
        y_lo, y_hi = 0, 1

    fig, axes = plt.subplots(2, 2, figsize=(3.46, 2.9))
    fig.suptitle(abbrev, fontsize=FS + 1.5, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.99, top=0.88, bottom=0.10,
                        hspace=0.45, wspace=0.18)

    for ri, mode in enumerate(MODES):
        for ci, dep_label in enumerate(DEP_LABELS):
            ax = axes[ri][ci]
            draw_panel_model(ax, panel_cache[(mode, dep_label)],
                             PANEL_TITLES[(mode, dep_label)],
                             show_ylabel=(ci == 0),
                             show_xlabel=(ri == 1),
                             label_fontsize=FS)
            ax.set_ylim(y_lo, y_hi)

    return fig


def build_combined_model_figure(records: list,
                                model_filter: Optional[set] = None) -> plt.Figure:
    """2×3 grid of per-model 2×2 panels with a shared N legend.

    Layout: each model row uses a GridSpec whose columns are
      [ci0, ci1, SPACER, ci0, ci1, SPACER, ci0, ci1]
    The spacer columns (half panel width) create a clear visual gap between
    model blocks without needing any extra axes.

    Panel titles are shown only in the top model row (mr==0).  Bottom rows
    get abbreviated "Block" / "Sparse" row-level labels instead.
    """
    import matplotlib.gridspec as gridspec

    MODEL_ORDER = [
        "Qwen3.5-27B-FP8", "Qwen3.5-9B", "gemini-3-flash-preview",
        "gemma-4-31B-it", "gpt-oss-120b", "gpt-oss-20b",
    ]
    all_in_data = sorted({r["model_slug"] for r in records
                          if r.get("completion_tokens", 0) > 0})
    if model_filter:
        all_in_data = [m for m in all_in_data if m in model_filter]
    models = [m for m in MODEL_ORDER if m in all_in_data]
    models += [m for m in all_in_data if m not in MODEL_ORDER]

    N_MCOLS  = 3
    N_MROWS  = (len(models) + N_MCOLS - 1) // N_MCOLS
    FS       = 5.5
    legend_h = 0.06
    title_h  = 0.055   # figure-fraction above each subplot block for model name
    gap_h    = 0.09    # generous gap between model rows

    total_h  = 1.0 - legend_h - (N_MROWS - 1) * gap_h
    block_h  = total_h / N_MROWS
    subplot_h = block_h - title_h

    # GridSpec column layout: panel | panel | spacer | panel | panel | spacer | panel | panel
    # spacer width = 0.55× panel width → clearly wider than wspace inside a model block
    SP = 0.55                                         # spacer ratio relative to a panel
    width_ratios = []
    for mc in range(N_MCOLS):
        width_ratios += [1, 1]
        if mc < N_MCOLS - 1:
            width_ratios.append(SP)
    N_SP_COLS = len(width_ratios)                    # 8 for N_MCOLS=3

    def gs_col(mc: int, ci: int) -> int:
        """Map (model-col, panel-col) to GridSpec column index."""
        return mc * (2 + 1) + ci                     # stride = 2 panels + 1 spacer

    MODE_LABELS = {0: "Block", 1: "Sparse"}

    fig = plt.figure(figsize=(7.2, N_MROWS * 3.6))
    legend_handles = legend_labels = None

    for mr in range(N_MROWS):
        from_bottom = N_MROWS - 1 - mr
        gs_bot = legend_h + from_bottom * (block_h + gap_h)
        gs_top = gs_bot + subplot_h
        y_name = gs_top + title_h * 0.5

        gs = gridspec.GridSpec(
            2, N_SP_COLS,
            left=0.07, right=0.995,
            bottom=gs_bot, top=gs_top,
            hspace=0.58, wspace=0.28,
            width_ratios=width_ratios,
        )

        for mc in range(N_MCOLS):
            midx = mr * N_MCOLS + mc
            if midx >= len(models):
                for ri in range(2):
                    for ci in range(2):
                        fig.add_subplot(gs[ri, gs_col(mc, ci)]).set_visible(False)
                continue

            model_slug = models[midx]
            abbrev = MODEL_ABBREV.get(model_slug, model_slug)

            panel_cache: dict = {}
            all_vals: list = []
            for mode in MODES:
                for dep_label in DEP_LABELS:
                    data = panel_data_for_model(records, mode, dep_label, model_slug)
                    panel_cache[(mode, dep_label)] = data
                    for n_dict in data.values():
                        all_vals.extend(n_dict.values())
            if all_vals:
                pad = (max(all_vals) - min(all_vals)) * 0.08
                y_lo = max(0.0, min(all_vals) - pad)
                y_hi = max(all_vals) + pad
            else:
                y_lo, y_hi = 0.0, 1.0

            tl_ax = tr_ax = None
            for ri, mode in enumerate(MODES):
                for ci, dep_label in enumerate(DEP_LABELS):
                    ax = fig.add_subplot(gs[ri, gs_col(mc, ci)])

                    # full panel titles only for the top model row; compact mode
                    # labels ("Block" / "Sparse") for all other rows
                    show_title = (mr == 0)
                    draw_panel_model(ax, panel_cache[(mode, dep_label)],
                                     PANEL_TITLES[(mode, dep_label)],
                                     show_title=show_title,
                                     show_ylabel=(ci == 0),
                                     show_xlabel=(ri == 1),
                                     label_fontsize=FS)
                    ax.set_ylim(y_lo, y_hi)
                    if ci == 0 and mc > 0:
                        ax.set_ylabel("")
                        ax.tick_params(axis="y", labelleft=True)
                    if ri == 0 and ci == 0:
                        tl_ax = ax
                    if ri == 0 and ci == 1:
                        tr_ax = ax
                    if legend_handles is None:
                        legend_handles, legend_labels = ax.get_legend_handles_labels()

                    # for lower model rows: compact mode label on left panel of each row
                    if mr > 0 and ci == 0:
                        ax.set_title(MODE_LABELS[ri], fontsize=FS, fontweight="bold",
                                     pad=3, loc="left")

            x_mid = (tl_ax.get_position().x0 + tr_ax.get_position().x1) / 2
            fig.text(x_mid, y_name, abbrev, ha="center", va="center",
                     fontsize=FS + 1.5, fontweight="bold")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=len(legend_handles), fontsize=FS + 0.5, frameon=False,
                   bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.2, handletextpad=0.3, columnspacing=0.6)
    return fig


def build_single_panel_fig(records: list, mode: str, dep_label: str,
                            n_filter: Optional[int] = None,
                            model_filter: Optional[set] = None,
                            label_fontsize: float = 11) -> Optional[plt.Figure]:
    """1×1 figure — no panel title, legend embedded inside axes."""
    data = panel_data(records, mode, dep_label, n_filter, model_filter)
    if not data:
        return None

    all_n: set = set()
    for m_dict in data.values():
        all_n.update(m_dict.keys())
    n_list = sorted(all_n)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.13)
    draw_panel(ax, data, title="", n_list=n_list,
               show_title=False, label_fontsize=label_fontsize)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=label_fontsize - 1,
              frameon=False, loc="best")
    return fig


# ── flat (1×1) builders ───────────────────────────────────────────────────────

def _flat_axes(ax, label_fontsize: float = 9) -> None:
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS], fontsize=label_fontsize)
    ax.set_xlabel("Window position", fontsize=label_fontsize)
    ax.set_ylabel("Mean completion tokens", fontsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def build_flat_model_fig(records: list, model_slug: str) -> plt.Figure:
    """Single-axes: mean completion tokens vs window, one line per N value.
    Averaged across all mode × dep conditions."""
    abbrev = MODEL_ABBREV.get(model_slug, model_slug)

    # group by n_options → window → list of tokens (all conditions pooled)
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["model_slug"] == model_slug and r.get("completion_tokens", 0) > 0:
            groups[r["n_options"]][r["window"]].append(r["completion_tokens"])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    fig.suptitle(abbrev, fontsize=11, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.13)

    for n_opt in sorted(groups):
        color  = N_COLORS.get(n_opt, "#333333")
        ls     = N_LINESTYLES.get(n_opt, "-")
        w_dict = groups[n_opt]
        xs = sorted(w_dict)
        ys = [float(np.mean(w_dict[x])) for x in xs]
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.0,
                marker="o", markersize=6, label=f"N={n_opt}", zorder=3)

    _flat_axes(ax)
    return fig


def build_flat_n_fig(records: list, n_val: int,
                     model_filter: Optional[set] = None) -> plt.Figure:
    """Single-axes: mean completion tokens vs window, one line per model.
    Averaged across all mode × dep conditions."""
    # group by model_slug → window → list of tokens (all conditions pooled)
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["n_options"] != n_val or r.get("completion_tokens", 0) <= 0:
            continue
        if model_filter and r["model_slug"] not in model_filter:
            continue
        groups[r["model_slug"]][r["window"]].append(r["completion_tokens"])

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    fig.suptitle(f"N = {n_val}", fontsize=11, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.13)

    for model_slug in sorted(groups):
        color  = MODEL_COLORS.get(model_slug, "#333333")
        abbrev = MODEL_ABBREV.get(model_slug, model_slug)
        w_dict = groups[model_slug]
        xs = sorted(w_dict)
        ys = [float(np.mean(w_dict[x])) for x in xs]
        ax.plot(xs, ys, color=color, linestyle="-", linewidth=2.0,
                marker="o", markersize=6, label=abbrev, zorder=3)

    _flat_axes(ax)
    return fig


# ── combined column figure (N=100 / 500 / 1000 stacked, one-column width) ─────

def build_triple_flat_column_fig(records: list,
                                  model_filter: Optional[set] = None) -> plt.Figure:
    """Three flat panels (N=100/500/1000) side by side, total width = one
    column of a two-column paper (~3.46 in).  One line per model, averaged
    across mode × dep conditions."""
    N_VALS = [100, 500, 1000]
    FS     = 5.5   # base font size — must stay readable at column width

    # collect per-N data
    all_groups: dict = {}
    for n_val in N_VALS:
        groups: dict = defaultdict(lambda: defaultdict(list))
        for r in records:
            if r["n_options"] != n_val or r.get("completion_tokens", 0) <= 0:
                continue
            if model_filter and r["model_slug"] not in model_filter:
                continue
            groups[r["model_slug"]][r["window"]].append(r["completion_tokens"])
        all_groups[n_val] = groups

    # compute global max across all N values to set a shared y ceiling
    global_max = max(
        float(np.mean(vals))
        for groups in all_groups.values()
        for w_dict in groups.values()
        for vals in w_dict.values()
    )
    y_ceil = int(np.ceil(global_max / 1000)) * 1000 + 300   # next 1K tick + 300 spare

    # 3.46" wide (one column), ~2.2" tall with legend above
    fig, axes = plt.subplots(1, 3, figsize=(3.46, 2.1))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.83, bottom=0.22,
                        wspace=0.18)

    for idx, n_val in enumerate(N_VALS):
        ax     = axes[idx]
        groups = all_groups[n_val]
        for model_slug in sorted(groups):
            color  = MODEL_COLORS.get(model_slug, "#333333")
            abbrev = MODEL_ABBREV.get(model_slug, model_slug)
            w_dict = groups[model_slug]
            xs = sorted(w_dict)
            ys = [float(np.mean(w_dict[x])) for x in xs]
            ax.plot(xs, ys, color=color, linestyle="-", linewidth=1.0,
                    marker="o", markersize=2.5, label=abbrev, zorder=3)

        ax.set_xticks(WINDOWS)
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=FS - 0.5)
        ax.set_xlabel("Window", fontsize=FS, labelpad=2)
        if idx == 0:
            ax.set_ylabel("Mean tokens", fontsize=FS, labelpad=2)
        else:
            ax.set_ylabel("")
        # shared y range, 1K steps
        ax.set_ylim(0, y_ceil)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda val, _: f"{int(val/1000)}K" if val >= 1000 else str(int(val))
            )
        )
        ax.tick_params(axis="y", labelsize=FS - 0.5,
                       labelleft=(idx == 0))
        ax.set_title(f"N = {n_val}", fontsize=FS + 0.5, fontweight="bold", pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, color="#DDDDDD", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)

    # shared legend centred above all three panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               ncol=3,
               fontsize=FS + 0.5, frameon=False,
               bbox_to_anchor=(0.565, 1.0),
               handlelength=1.2, handletextpad=0.3, columnspacing=0.6)
    return fig


# ── flat legend figures ───────────────────────────────────────────────────────

def build_flat_legend_model() -> plt.Figure:
    """Horizontal legend: one entry per N value (for per-model flat plots)."""
    handles = [
        mlines.Line2D([], [], color=N_COLORS[n], linestyle=N_LINESTYLES[n],
                      linewidth=2, marker="o", markersize=6, label=f"N={n}")
        for n in sorted(N_COLORS)
    ]
    fig, ax = plt.subplots(figsize=(4, 0.45))
    ax.set_axis_off()
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               fontsize=11, frameon=False)
    return fig


def build_flat_legend_n(model_filter: Optional[set] = None) -> plt.Figure:
    """Horizontal legend: one entry per model (for per-N flat plots)."""
    slugs = [s for s in MODEL_ABBREV if model_filter is None or s in model_filter]
    handles = [
        mlines.Line2D([], [], color=MODEL_COLORS.get(s, "#333333"), linestyle="-",
                      linewidth=2, marker="o", markersize=6,
                      label=MODEL_ABBREV[s])
        for s in slugs
    ]
    fig, ax = plt.subplots(figsize=(len(handles) * 1.4, 0.45))
    ax.set_axis_off()
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               fontsize=11, frameon=False)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    records = load_records(args.records_path)

    model_filter = None
    if args.models:
        model_filter = resolve_model_slugs([m.strip() for m in args.models.split(",")])
        print(f"Model filter: {sorted(model_filter)}")

    n_filter = args.n_options

    base_out = Path(args.output)
    out_dir  = base_out.parent / "reasoning_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem   = base_out.stem
    suffix = base_out.suffix

    # flat legends
    fig = build_flat_legend_model()
    fig.savefig(out_dir / f"{stem}_legend_flat_model{suffix}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_dir / f'{stem}_legend_flat_model{suffix}'}")

    fig = build_flat_legend_n(model_filter)
    fig.savefig(out_dir / f"{stem}_legend_flat_N{suffix}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_dir / f'{stem}_legend_flat_N{suffix}'}")

    # overall 2×2
    fig = build_figure(records, n_filter=n_filter, model_filter=model_filter)
    p = out_dir / f"{stem}{suffix}"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    # combined one-column figure (N=100/500/1000 stacked)
    fig = build_triple_flat_column_fig(records, model_filter)
    p   = out_dir / f"{stem}_triple_column{suffix}"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {p}")

    # combined 2×3 model grid
    fig = build_combined_model_figure(records, model_filter)
    p   = out_dir / f"{stem}_model_grid{suffix}"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {p}")

    # per-N: 2×2 + four 1×1 panels + flat single-axes
    all_n = sorted({r["n_options"] for r in records})
    for n_val in all_n:
        fig = build_figure(records, n_filter=n_val, model_filter=model_filter)
        n_path = out_dir / f"{stem}_N{n_val}{suffix}"
        fig.savefig(n_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {n_path}")

        fig = build_flat_n_fig(records, n_val, model_filter)
        p = out_dir / f"{stem}_N{n_val}_flat{suffix}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")

        for mode in MODES:
            for dep_label in DEP_LABELS:
                dep_slug   = dep_label.replace("option_", "")
                panel_path = out_dir / f"{stem}_N{n_val}_{mode}_{dep_slug}{suffix}"
                fig = build_single_panel_fig(records, mode, dep_label,
                                              n_val, model_filter)
                if fig is not None:
                    fig.savefig(panel_path, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    print(f"Saved → {panel_path}")

    # per-model 2×2 figures (lines = N values)
    all_models = sorted({r["model_slug"] for r in records
                         if r.get("completion_tokens", 0) > 0})
    if model_filter is not None:
        all_models = [m for m in all_models if m in model_filter]

    for model_slug in all_models:
        abbrev = MODEL_ABBREV.get(model_slug, model_slug).replace(" ", "_")
        fig    = build_model_figure(records, model_slug)
        p      = out_dir / f"{stem}_model_{abbrev}{suffix}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")

        fig = build_flat_model_fig(records, model_slug)
        p   = out_dir / f"{stem}_model_{abbrev}_flat{suffix}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-path", required=True,
                        help="Path to reasoning_records.jsonl produced by analyze_reasoning.py")
    parser.add_argument("--output", default=None,
                        help="Base output path (default: out/reasoning_2x2.pdf)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated model names/slugs (default: all). "
                             "E.g. 'Qwen3.5-27B,GPT-OSS-120B,Gemini-3-Flash'")
    parser.add_argument("--n-options", type=int, default=None,
                        help="Restrict to one N value (default: all)")
    args = parser.parse_args()
    if args.output is None:
        args.output = "out/reasoning_2x2.pdf"
    main(args)
