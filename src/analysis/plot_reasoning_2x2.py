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
                     label_fontsize: float = 8) -> None:
    """Draw one panel for a single model — one line per N value, colored by N."""
    for n_opt in sorted(data):
        color  = N_COLORS.get(n_opt, "#333333")
        ls     = N_LINESTYLES.get(n_opt, "-")
        w_dict = data[n_opt]
        xs = sorted(w_dict)
        ys = [w_dict[x] for x in xs]
        ax.plot(xs, ys, color=color, linestyle=ls, linewidth=2.0,
                marker="o", markersize=6, label=f"N={n_opt}", zorder=3)

    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS], fontsize=label_fontsize)
    if show_xlabel:
        ax.set_xlabel("Window position", fontsize=label_fontsize)
    if show_ylabel:
        ax.set_ylabel("Mean completion tokens", fontsize=label_fontsize)
    ax.tick_params(axis="y", labelsize=label_fontsize - 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    if show_title:
        ax.set_title(title, fontsize=9, fontweight="bold", pad=5)


def build_model_figure(records: list, model_slug: str) -> plt.Figure:
    """2×2 figure for a single model — lines = N values (100 / 500 / 1000)."""
    abbrev = MODEL_ABBREV.get(model_slug, model_slug)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(abbrev, fontsize=12, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.13,
                        hspace=0.22, wspace=0.15)

    for ri, mode in enumerate(MODES):
        for ci, dep_label in enumerate(DEP_LABELS):
            data = panel_data_for_model(records, mode, dep_label, model_slug)
            draw_panel_model(axes[ri][ci], data, PANEL_TITLES[(mode, dep_label)],
                             show_ylabel=(ci == 0),
                             show_xlabel=(ri == 1))

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
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
