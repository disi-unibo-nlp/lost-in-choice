"""
Two complementary figures comparing completion-token distributions for
correct vs. incorrect predictions (sparse mode).

  tokens_violin.pdf     — 2×3 grid of violin plots (one per model).
                          Each panel: correct (left) vs. wrong (right) violin,
                          log y-axis.  Shows full distribution shape per model.

  tokens_violin_legend.pdf — standalone legend

  tokens_cdf.pdf        — single panel: CDF curves for correct vs. wrong,
                          one line per model, log x-axis.  Horizontal gap
                          between curves = median token surplus for wrong answers.

  tokens_cdf_legend.pdf — standalone legend

Usage:
    python src/analysis/plot_tokens_correct_vs_wrong.py \\
        --records-path out/reasoning_analysis/reasoning_records.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

MODEL_ABBREV = {
    "Qwen3.5-27B-FP8":        "Qwen3.5-27B",
    "Qwen3.5-9B":             "Qwen3.5-9B",
    "gemini-3-flash-preview":  "Gemini-3-Flash",
    "gemma-4-31B-it":         "Gemma-4-31B",
    "gpt-oss-120b":           "GPT-OSS-120B",
    "gpt-oss-20b":            "GPT-OSS-20B",
}

CORRECT_COLOR = "#3182BD"   # blue
WRONG_COLOR   = "#E6550D"   # orange-red

# Per-model line colours for the CDF figure
MODEL_COLORS = [
    "#1F77B4", "#FF7F0E", "#2CA02C",
    "#D62728", "#9467BD", "#8C564B",
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_records(path: str, mode_filter: str = "sparse") -> list:
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if (r.get("completion_tokens") or 0) <= 0:
                continue
            if mode_filter and r.get("mode") != mode_filter:
                continue
            records.append(r)
    return records


# ── violin figure ─────────────────────────────────────────────────────────────

def build_violin_figure(records: list) -> plt.Figure:
    """2×3 grid: one panel per model, split violin correct/wrong."""
    models = sorted(
        {r["model_slug"] for r in records},
        key=lambda m: MODEL_ABBREV.get(m, m),
    )
    n_models = len(models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if n_models > 1 else [axes]
    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.10,
                        hspace=0.38, wspace=0.32)

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        recs = [r for r in records if r["model_slug"] == model]
        correct_toks = [r["completion_tokens"] for r in recs if r["correct"]]
        wrong_toks   = [r["completion_tokens"] for r in recs if not r["correct"]]

        # log10 values for violin (matplotlib violinplot works on raw arrays)
        log_c = np.log10(correct_toks)
        log_w = np.log10(wrong_toks)

        for log_vals, pos, color in [
            (log_c, 1, CORRECT_COLOR),
            (log_w, 2, WRONG_COLOR),
        ]:
            parts = ax.violinplot(
                [log_vals],
                positions=[pos],
                widths=0.6,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.70)
                pc.set_edgecolor("none")
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(1.8)

        # y-axis: log10 ticks → actual token labels
        all_log = np.concatenate([log_c, log_w])
        ymin = max(0, np.floor(all_log.min()) - 0.2)
        ymax = np.ceil(all_log.max()) + 0.2
        ax.set_ylim(ymin, ymax)

        tick_powers = [p for p in range(int(ymin), int(ymax) + 1)]
        ax.set_yticks(tick_powers)
        ax.set_yticklabels(
            [f"$10^{{{p}}}$" for p in tick_powers], fontsize=7.5
        )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Correct", "Wrong"], fontsize=9)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(0.4, 2.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)

        abbrev = MODEL_ABBREV.get(model, model)
        ax.text(0.5, 0.97, abbrev, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, fontweight="bold")

        if idx % ncols == 0:
            ax.set_ylabel("Completion tokens", fontsize=8.5)

    # hide unused panels
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    return fig


def build_violin_legend() -> plt.Figure:
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CORRECT_COLOR, alpha=0.7, label="Correct"),
        plt.Rectangle((0, 0), 1, 1, facecolor=WRONG_COLOR,   alpha=0.7, label="Wrong"),
    ]
    fig, ax = plt.subplots(figsize=(2.5, 0.5))
    ax.set_axis_off()
    fig.legend(handles, [h.get_label() for h in handles],
               loc="center", ncol=2, fontsize=9, frameon=False)
    return fig


# ── CDF figure ────────────────────────────────────────────────────────────────

def build_cdf_figure(records: list) -> plt.Figure:
    """Single panel: CDF of completion_tokens, one pair of lines per model."""
    models = sorted(
        {r["model_slug"] for r in records},
        key=lambda m: MODEL_ABBREV.get(m, m),
    )

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.12)

    for model, color in zip(models, MODEL_COLORS):
        recs = [r for r in records if r["model_slug"] == model]
        correct_toks = sorted(r["completion_tokens"] for r in recs if r["correct"])
        wrong_toks   = sorted(r["completion_tokens"] for r in recs if not r["correct"])
        abbrev       = MODEL_ABBREV.get(model, model)

        for toks, ls, lw, label in [
            (correct_toks, "-",  1.6, f"{abbrev} — correct"),
            (wrong_toks,   "--", 1.6, f"{abbrev} — wrong"),
        ]:
            n   = len(toks)
            cdf = np.arange(1, n + 1) / n * 100
            ax.semilogx(toks, cdf, color=color, linestyle=ls,
                        linewidth=lw, label=label, zorder=3)

    ax.set_xlabel("Completion tokens (log scale)", fontsize=10)
    ax.set_ylabel("Cumulative fraction (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 102)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)

    return fig


def build_cdf_legend(models: list) -> plt.Figure:
    handles = []
    for model, color in zip(models, MODEL_COLORS):
        abbrev = MODEL_ABBREV.get(model, model)
        handles.append(plt.Line2D([0], [0], color=color, linestyle="-",
                                   linewidth=1.6, label=f"{abbrev} — correct"))
        handles.append(plt.Line2D([0], [0], color=color, linestyle="--",
                                   linewidth=1.6, label=f"{abbrev} — wrong"))
    ncols = len(models)
    fig, ax = plt.subplots(figsize=(ncols * 2.0, 0.9))
    ax.set_axis_off()
    fig.legend(handles, [h.get_label() for h in handles],
               loc="center", ncol=2, fontsize=8, frameon=False)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    records = load_records(args.records_path, mode_filter="sparse")
    print(f"Loaded {len(records):,} sparse records with token info")

    models = sorted(
        {r["model_slug"] for r in records},
        key=lambda m: MODEL_ABBREV.get(m, m),
    )

    # quick summary
    print(f"\n{'Model':<28}  {'correct med':>11}  {'wrong med':>9}  {'ratio':>6}")
    for m in models:
        recs = [r for r in records if r["model_slug"] == m]
        c = sorted(r["completion_tokens"] for r in recs if r["correct"])
        w = sorted(r["completion_tokens"] for r in recs if not r["correct"])
        if not c or not w: continue
        print(f"  {MODEL_ABBREV.get(m,m):<26}  {c[len(c)//2]:>11,}  "
              f"{w[len(w)//2]:>9,}  {w[len(w)//2]/c[len(c)//2]:>6.2f}x")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_violin_figure(records)
    p   = out_dir / "tokens_violin.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {p}")

    fig = build_violin_legend()
    p   = out_dir / "tokens_violin_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_cdf_figure(records)
    p   = out_dir / "tokens_cdf.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_cdf_legend(models)
    p   = out_dir / "tokens_cdf_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot token distributions: correct vs. wrong (sparse mode)."
    )
    parser.add_argument(
        "--records-path",
        default="out/reasoning_analysis/reasoning_records.jsonl",
        help="Path to reasoning_records.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="out/figures/tokens_correct_vs_wrong",
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args)
