"""
Error rate vs. completion-token load (sparse mode).

For each model independently, records are ranked by completion_tokens and split
into deciles (0 = lightest 10 %, 9 = heaviest 10 %).  Error rate is then
computed per (decile, N-options) cell, pooled across all models.

Within-model decile normalisation removes the confound that some models are
simply harder than others and also tend to produce longer chains.

Produces:
  error_vs_tokens.pdf          — error rate per decile, one line per N
  error_vs_tokens_legend.pdf   — standalone legend

Usage:
    python src/analysis/plot_error_vs_tokens.py \\
        --records-path out/reasoning_analysis/reasoning_records.jsonl

    # restrict to specific N values:
    python src/analysis/plot_error_vs_tokens.py \\
        --records-path ... --n-options 500 1000
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── config ────────────────────────────────────────────────────────────────────

MODEL_ABBREV_MAP = {
    "Qwen3.5-27B-FP8":        "Qwen3.5-27B",
    "Qwen3.5-9B":             "Qwen3.5-9B",
    "gemini-3-flash-preview":  "Gemini-3-Flash",
    "gemma-4-31B-it":         "Gemma-4-31B",
    "gpt-oss-120b":           "GPT-OSS-120B",
    "gpt-oss-20b":            "GPT-OSS-20B",
}

MODEL_COLORS_MAP = {
    "Qwen3.5-27B-FP8":        "#4C72B0",
    "Qwen3.5-9B":             "#DD8452",
    "gemini-3-flash-preview":  "#55A868",
    "gemma-4-31B-it":         "#C44E52",
    "gpt-oss-120b":           "#8172B2",
    "gpt-oss-20b":            "#937860",
}

N_PALETTE = {
    100:  "#9ECAE1",
    500:  "#3182BD",
    1000: "#08306B",
}
N_MARKERS = {100: "o", 500: "s", 1000: "^"}

N_DECILES = 10


# ── data helpers ──────────────────────────────────────────────────────────────

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


def assign_within_model_deciles(records: list) -> None:
    """Add 'token_decile' (0–9) in-place, ranked within each model."""
    by_model: Dict[str, list] = defaultdict(list)
    for r in records:
        by_model[r["model_slug"]].append(r)

    for recs in by_model.values():
        sorted_toks = sorted(r["completion_tokens"] for r in recs)
        n = len(sorted_toks)
        # percentile thresholds: 10th, 20th, ..., 90th
        thresholds = [sorted_toks[min(n - 1, int((p / 10) * n))]
                      for p in range(1, N_DECILES)]
        for r in recs:
            t = r["completion_tokens"]
            decile = 0
            for thr in thresholds:
                if t > thr:
                    decile += 1
                else:
                    break
            r["token_decile"] = decile


def compute_error_curves(
    records: list, n_options_filter: Optional[List[int]] = None
) -> Dict[int, Dict[int, float]]:
    """
    Returns {n_options: {decile: error_rate_pct}}.
    Pooled across all models (within-model deciles already assigned).
    """
    if n_options_filter:
        records = [r for r in records if r["n_options"] in n_options_filter]

    all_n = sorted({r["n_options"] for r in records})
    result: Dict[int, Dict[int, float]] = {}

    for n in all_n:
        subset = [r for r in records if r["n_options"] == n]
        by_decile: Dict[int, list] = defaultdict(list)
        for r in subset:
            by_decile[r["token_decile"]].append(r)
        result[n] = {}
        for d in range(N_DECILES):
            recs = by_decile.get(d, [])
            if recs:
                result[n][d] = 100 * sum(1 for r in recs if not r["correct"]) / len(recs)

    return result


def mean_tokens_per_decile(records: list) -> Dict[int, float]:
    """Mean completion_tokens per decile (pooled across all N and models)."""
    by_decile: Dict[int, list] = defaultdict(list)
    for r in records:
        by_decile[r["token_decile"]].append(r["completion_tokens"])
    return {d: sum(v) / len(v) for d, v in by_decile.items() if v}


# ── drawing ───────────────────────────────────────────────────────────────────

def build_figure(
    curves: Dict[int, Dict[int, float]],
    mean_toks: Dict[int, float],
) -> plt.Figure:
    all_n = sorted(curves)
    xs    = list(range(N_DECILES))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.12)

    for n in all_n:
        pts = curves[n]
        ys  = [pts.get(d) for d in xs]
        ax.plot(xs, ys,
                color=N_PALETTE.get(n, "#333333"),
                marker=N_MARKERS.get(n, "o"),
                markersize=5, linewidth=2.0, zorder=3,
                label=f"N = {n:,}")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d*10}–{d*10+10}" for d in xs],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_xlabel("Completion-token percentile (within model, %)", fontsize=9)
    ax.set_ylabel("Error rate (%)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9)

    return fig


def build_legend(all_n: List[int]) -> plt.Figure:
    handles = [
        plt.Line2D([0], [0],
                   color=N_PALETTE.get(n, "#333333"),
                   marker=N_MARKERS.get(n, "o"),
                   markersize=5, linewidth=2.0,
                   label=f"N = {n:,}")
        for n in all_n
    ]
    fig, ax = plt.subplots(figsize=(3.5, 0.5))
    ax.set_axis_off()
    fig.legend(handles, [h.get_label() for h in handles],
               loc="center", ncol=len(handles), fontsize=9, frameon=False)
    return fig


# ── combined one-column figure ────────────────────────────────────────────────

def build_combined_figure(records: list,
                           curves: Dict[int, Dict[int, float]],
                           mean_toks: Dict[int, float]) -> plt.Figure:
    """3-panel one-column figure (3.46"):
       panel 0 (60%) — error rate vs token percentile (existing plot)
       panel 1 (20%) — mean tokens when correct, x=N, lines=model
       panel 2 (20%) — mean tokens when wrong,   x=N, lines=model
    """
    FS     = 5.5
    N_VALS = [100, 500, 1000]

    # per-model × N × correctness
    by_model: Dict = defaultdict(lambda: defaultdict(lambda: {"correct": [], "wrong": []}))
    for r in records:
        key = "correct" if r["correct"] else "wrong"
        by_model[r["model_slug"]][r["n_options"]][key].append(r["completion_tokens"])

    models = sorted(by_model.keys(), key=lambda m: MODEL_ABBREV_MAP.get(m, m))

    # shared y-range for panels 1 & 2
    all_means = [
        float(np.mean(by_model[m][n][k]))
        for m in models for n in N_VALS for k in ("correct", "wrong")
        if by_model[m][n][k]
    ]
    y_ceil = int(np.ceil(max(all_means) / 1000)) * 1000 + 300

    fig, axes = plt.subplots(1, 3, figsize=(3.46, 2.3),
                              gridspec_kw={"width_ratios": [3, 1, 1],
                                           "wspace": 0.40})
    # sub-GridSpec trick: tighten gap between panels 1 & 2 via a nested spec
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_outer = fig.add_gridspec(1, 2, left=0.12, right=0.99,
                                top=0.70, bottom=0.22,
                                width_ratios=[3, 2], wspace=0.36)
    gs_right = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1], wspace=0.30)
    plt.close(fig)  # discard preliminary figure

    fig = plt.figure(figsize=(3.46, 2.3))
    ax0 = fig.add_subplot(gs_outer[0])
    ax1 = fig.add_subplot(gs_right[0])
    ax2 = fig.add_subplot(gs_right[1])
    axes = [ax0, ax1, ax2]

    # ── panel 0 : error vs token percentile ──────────────────────────────────
    ax    = axes[0]
    xs    = list(range(N_DECILES))
    all_n = sorted(curves)
    for n in all_n:
        pts = curves[n]
        ys  = [pts.get(d) for d in xs]
        ax.plot(xs, ys, color=N_PALETTE.get(n, "#333333"),
                marker=N_MARKERS.get(n, "o"), markersize=2.5,
                linewidth=0.9, zorder=3)

    tick_idx = list(range(0, N_DECILES, 2))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{d*10}–{d*10+10}" for d in tick_idx],
                       rotation=0, ha="center", fontsize=FS - 1)
    ax.set_xlabel("Token percentile", fontsize=FS, labelpad=1)
    ax.set_ylabel("Error rate", fontsize=FS, labelpad=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS - 0.5)

    # ── panels 1 & 2 : mean tokens correct / wrong ───────────────────────────
    x_pos = [0, 1, 2]
    k_fmt = mticker.FuncFormatter(
        lambda val, _: f"{int(val/1000)}K" if val >= 1000 else str(int(val))
    )

    for pi, ckey in enumerate(("correct", "wrong")):
        ax = axes[pi + 1]
        for model in models:
            color = MODEL_COLORS_MAP.get(model, "#333333")
            ys = [
                float(np.mean(by_model[model][n][ckey]))
                if by_model[model][n][ckey] else float("nan")
                for n in N_VALS
            ]
            ax.plot(x_pos, ys, color=color, linestyle="-", linewidth=0.9,
                    marker="o", markersize=2.5, zorder=3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["100", "500", "1K"], fontsize=FS - 0.5)
        ax.set_xlabel("N", fontsize=FS, labelpad=2)
        ax.set_ylim(0, y_ceil)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
        ax.yaxis.set_major_formatter(k_fmt)
        ax.set_title("Correct" if ckey == "correct" else "Wrong",
                     fontsize=FS + 0.5, fontweight="bold", pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        if pi == 0:
            ax.set_ylabel("Mean tokens", fontsize=FS, labelpad=2)
            ax.tick_params(axis="y", labelsize=FS - 0.5)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # ── single shared legend: top-centre, 3 cols × 3 rows (N values + models) ─
    n_handles = [
        plt.Line2D([0], [0], color=N_PALETTE[n], marker=N_MARKERS[n],
                   markersize=2.5, linewidth=0.9, label=f"N={n:,}")
        for n in all_n
    ]
    m_handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS_MAP.get(m, "#333333"),
                   linestyle="-", linewidth=0.9, marker="o", markersize=2.5,
                   label=MODEL_ABBREV_MAP.get(m, m))
        for m in models
    ]
    all_handles = n_handles + m_handles   # 3 + 6 = 9 items → 3×3
    fig.legend(all_handles, [h.get_label() for h in all_handles],
               loc="upper center", ncol=3,
               fontsize=FS + 0.5, frameon=False,
               bbox_to_anchor=(0.565, 1.0),
               handlelength=1.0, handletextpad=0.3,
               columnspacing=0.5, labelspacing=0.15)

    return fig


def build_tokens_panel_fig(records: list, correctness_key: str) -> plt.Figure:
    """Standalone figure for the correct or wrong tokens panel."""
    FS     = 5.5
    N_VALS = [100, 500, 1000]

    by_model: Dict = defaultdict(lambda: defaultdict(lambda: {"correct": [], "wrong": []}))
    for r in records:
        key = "correct" if r["correct"] else "wrong"
        by_model[r["model_slug"]][r["n_options"]][key].append(r["completion_tokens"])

    models = sorted(by_model.keys(), key=lambda m: MODEL_ABBREV_MAP.get(m, m))

    all_means = [
        float(np.mean(by_model[m][n][k]))
        for m in models for n in N_VALS for k in ("correct", "wrong")
        if by_model[m][n][k]
    ]
    y_ceil = int(np.ceil(max(all_means) / 1000)) * 1000 + 300

    fig, ax = plt.subplots(1, 1, figsize=(1.15, 1.8))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.83, bottom=0.22)

    k_fmt = mticker.FuncFormatter(
        lambda val, _: f"{int(val/1000)}K" if val >= 1000 else str(int(val))
    )
    x_pos = [0, 1, 2]
    for model in models:
        color  = MODEL_COLORS_MAP.get(model, "#333333")
        abbrev = MODEL_ABBREV_MAP.get(model, model)
        ys = [
            float(np.mean(by_model[model][n][correctness_key]))
            if by_model[model][n][correctness_key] else float("nan")
            for n in N_VALS
        ]
        ax.plot(x_pos, ys, color=color, linestyle="-", linewidth=0.9,
                marker="o", markersize=2.5, label=abbrev, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(["100", "500", "1K"], fontsize=FS - 0.5)
    ax.set_xlabel("N", fontsize=FS, labelpad=2)
    ax.set_ylabel("Mean tokens", fontsize=FS, labelpad=2)
    ax.set_ylim(0, y_ceil)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
    ax.yaxis.set_major_formatter(k_fmt)
    ax.tick_params(axis="y", labelsize=FS - 0.5)
    title = "Correct" if correctness_key == "correct" else "Wrong"
    ax.set_title(title, fontsize=FS + 0.5, fontweight="bold", pad=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    return fig


# ── separate component figures ────────────────────────────────────────────────

def _make_n_m_handles(all_n: list, models: list) -> list:
    n_handles = [
        plt.Line2D([0], [0], color=N_PALETTE[n], marker=N_MARKERS[n],
                   markersize=2.5, linewidth=0.9, label=f"N={n:,}")
        for n in all_n
    ]
    m_handles = [
        plt.Line2D([0], [0], color=MODEL_COLORS_MAP.get(m, "#333333"),
                   linestyle="-", linewidth=0.9, marker="o", markersize=2.5,
                   label=MODEL_ABBREV_MAP.get(m, m))
        for m in models
    ]
    return n_handles + m_handles


def build_left_only_fig(curves: Dict[int, Dict[int, float]]) -> plt.Figure:
    """Left panel only: error rate vs token percentile."""
    FS    = 5.5
    all_n = sorted(curves)
    xs    = list(range(N_DECILES))

    fig, ax = plt.subplots(1, 1, figsize=(2.1, 1.8))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.97, bottom=0.24)

    for n in all_n:
        pts = curves[n]
        ys  = [pts.get(d) for d in xs]
        ax.plot(xs, ys, color=N_PALETTE.get(n, "#333333"),
                marker=N_MARKERS.get(n, "o"), markersize=2.5,
                linewidth=0.9, zorder=3)

    tick_idx = list(range(0, N_DECILES, 2))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{d*10}–{d*10+10}" for d in tick_idx],
                       rotation=0, ha="center", fontsize=FS - 1)
    ax.set_xlabel("Token percentile", fontsize=FS, labelpad=1)
    ax.set_ylabel("Error rate", fontsize=FS, labelpad=2)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(0, 75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS - 0.5)
    return fig


def build_correct_wrong_fig(records: list) -> plt.Figure:
    """Correct + Wrong panels side by side, compact, no legend."""
    FS     = 5.5
    N_VALS = [100, 500, 1000]

    by_model: Dict = defaultdict(lambda: defaultdict(lambda: {"correct": [], "wrong": []}))
    for r in records:
        key = "correct" if r["correct"] else "wrong"
        by_model[r["model_slug"]][r["n_options"]][key].append(r["completion_tokens"])

    models = sorted(by_model.keys(), key=lambda m: MODEL_ABBREV_MAP.get(m, m))
    all_means = [
        float(np.mean(by_model[m][n][k]))
        for m in models for n in N_VALS for k in ("correct", "wrong")
        if by_model[m][n][k]
    ]
    y_ceil = int(np.ceil(max(all_means) / 1000)) * 1000 + 300
    k_fmt  = mticker.FuncFormatter(
        lambda val, _: f"{int(val/1000)}K" if val >= 1000 else str(int(val))
    )

    fig, axes = plt.subplots(1, 2, figsize=(1.5, 1.8))
    fig.subplots_adjust(left=0.22, right=0.99, top=0.88, bottom=0.18, wspace=0.30)

    x_pos = [0, 1, 2]
    for pi, ckey in enumerate(("correct", "wrong")):
        ax = axes[pi]
        for model in models:
            color = MODEL_COLORS_MAP.get(model, "#333333")
            ys = [
                float(np.mean(by_model[model][n][ckey]))
                if by_model[model][n][ckey] else float("nan")
                for n in N_VALS
            ]
            ax.plot(x_pos, ys, color=color, linestyle="-", linewidth=0.9,
                    marker="o", markersize=2.5, zorder=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["100", "500", "1K"], fontsize=FS - 0.5)
        ax.set_xlabel("N", fontsize=FS, labelpad=2)
        ax.set_ylim(0, y_ceil)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1000))
        ax.yaxis.set_major_formatter(k_fmt)
        ax.set_title("Correct" if ckey == "correct" else "Wrong",
                     fontsize=FS + 0.5, fontweight="bold", pad=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        if pi == 0:
            ax.set_ylabel("Mean tokens", fontsize=FS, labelpad=2)
            ax.tick_params(axis="y", labelsize=FS - 0.5)
        else:
            ax.tick_params(axis="y", labelleft=False)
    return fig


def build_combined_legend_fig(all_n: list, models: list) -> plt.Figure:
    """Standalone 3×3 legend (N values + models)."""
    FS      = 6.0
    handles = _make_n_m_handles(all_n, models)
    fig, ax = plt.subplots(figsize=(3.46, 0.75))
    ax.set_axis_off()
    fig.legend(handles, [h.get_label() for h in handles],
               loc="center", ncol=3,
               fontsize=FS, frameon=False,
               handlelength=1.0, handletextpad=0.3,
               columnspacing=0.5, labelspacing=0.2)
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    records = load_records(args.records_path, mode_filter="sparse")
    print(f"Loaded {len(records):,} sparse records with token info")

    n_filter = args.n_options if args.n_options else None
    assign_within_model_deciles(records)

    curves    = compute_error_curves(records, n_filter)
    mean_toks = mean_tokens_per_decile(records)
    all_n     = sorted(curves)

    print("\nError rate per token decile (sparse, pooled across models):")
    header = f"  {'decile':<8}" + "".join(f"  N={n:<6}" for n in all_n)
    print(header)
    for d in range(N_DECILES):
        tok_str = f"~{mean_toks.get(d,0):.0f} tok"
        row = f"  {d:<8}" + "".join(
            f"  {curves[n].get(d, float('nan')):>5.1f}%  " for n in all_n
        )
        print(row + f"  ({tok_str})")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = build_figure(curves, mean_toks)
    p   = out_dir / "error_vs_tokens.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {p}")

    fig = build_legend(all_n)
    p   = out_dir / "error_vs_tokens_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_combined_figure(records, curves, mean_toks)
    p   = out_dir / "error_vs_tokens_combined.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {p}")

    for ckey in ("correct", "wrong"):
        fig = build_tokens_panel_fig(records, ckey)
        p   = out_dir / f"tokens_{ckey}.pdf"
        fig.savefig(p, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"Saved → {p}")

    models = sorted({r["model_slug"] for r in records},
                    key=lambda m: MODEL_ABBREV_MAP.get(m, m))

    fig = build_left_only_fig(curves)
    p   = out_dir / "error_vs_tokens_left.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_correct_wrong_fig(records)
    p   = out_dir / "error_vs_tokens_right.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_combined_legend_fig(all_n, models)
    p   = out_dir / "error_vs_tokens_legend_combined.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot error rate vs completion-token decile (sparse mode)."
    )
    parser.add_argument(
        "--records-path",
        default="out/reasoning_analysis/reasoning_records.jsonl",
        help="Path to reasoning_records.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="out/figures/error_vs_tokens",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--n-options", type=int, nargs="+", default=None,
        help="Restrict to specific N values (default: all).",
    )
    args = parser.parse_args()
    main(args)
