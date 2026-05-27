"""
Positional bias figures: wrong picks before vs after the gold option.

Produces two figures:

  before_gold_main.pdf   (MAIN PAPER)
      Sparse mode only. Deviation from empirical null (pp) vs window position.
      One line per N (100 / 500 / 1000).  Caption argument: sparse mode isolates
      positional bias because the 9 wrong originals are distributed uniformly
      across segments, so option-type preference does not confound position.

  before_gold_appendix.pdf   (APPENDIX)
      Two panels side-by-side, same y-axis scale:
        Left  — sparse deviation (same data as main figure)
        Right — block deviation
      Block extreme-window deviations expose a second effect (preference for
      original options within the local block) distinct from positional bias.

Deviation is computed against the empirical null:
      null_pct = mean( gold_index / (n_options − 1) × 100 )
  averaged over wrong records in each (mode, window, N) cell.  For N ≥ 500 this
  is virtually identical to window × 100 %, but it is exact by construction.

Usage:
    python src/analysis/plot_before_gold.py \\
        --classified-path out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl

    # custom output directory:
    python src/analysis/plot_before_gold.py \\
        --classified-path ... --output figures/before_gold
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── constants ─────────────────────────────────────────────────────────────────

WINDOWS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Sequential palette for N values (light → dark)
N_PALETTE = {
    100:  "#9ECAE1",   # light blue
    500:  "#3182BD",   # medium blue
    1000: "#08306B",   # dark blue
}
N_MARKERS = {100: "o", 500: "s", 1000: "^"}

_CODE_RE = re.compile(r"^[Ss](\d+)")
_DATASET_TAG_RE = re.compile(
    r"mmlu-pro-augmented-(?P<n_options>\d+)__seed_(?P<seed>\d+)"
    r"__window_(?P<window>[\d.]+)__(?P<mode>block|sparse)"
)


# ── data loading ──────────────────────────────────────────────────────────────

def load_records_from_classified(path: str) -> list:
    """Load from classified_traces.jsonl (strategy-classified subset)."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    records = [r for r in records if r.get("confidence") in ("high", "medium")]
    for r in records:
        m = _CODE_RE.match(r["primary_strategy"].strip())
        if m:
            r["primary_strategy"] = f"S{m.group(1)}"
    return records


def load_records_from_completions(completions_dir: str) -> list:
    """
    Walk out/completions/<model>/<dataset_tag>/<timestamp>/predictions.jsonl
    and return flat list of records with fields:
        model_slug, n_options, window, mode,
        predicted_index, gold_index, correct.
    When multiple timestamps exist for the same (model, dataset_tag),
    the latest is used.
    """
    root = Path(completions_dir)
    latest: Dict[Tuple[str, str], Path] = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for ds_dir in sorted(model_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            if not _DATASET_TAG_RE.match(ds_dir.name):
                continue
            key = (model_dir.name, ds_dir.name)
            for ts_dir in sorted(d for d in ds_dir.iterdir() if d.is_dir()):
                if (ts_dir / "predictions.jsonl").exists():
                    latest[key] = ts_dir

    records = []
    for (model_slug, tag), run_dir in sorted(latest.items()):
        m = _DATASET_TAG_RE.match(tag)
        n_options = int(m.group("n_options"))
        window    = float(m.group("window"))
        mode      = m.group("mode")

        with open(run_dir / "predictions.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("predicted_index") is None or r.get("gold_index") is None:
                    continue
                records.append({
                    "model_slug":      model_slug,
                    "n_options":       n_options,
                    "window":          window,
                    "mode":            mode,
                    "predicted_index": r["predicted_index"],
                    "gold_index":      r["gold_index"],
                    "correct":         r.get("correct", False),
                    "option_dep":      r.get("option_dependent_vs_independent", "unknown"),
                })

    print(f"Loaded {len(records):,} predictions from {len(latest)} runs")
    return records


# ── core computation ──────────────────────────────────────────────────────────

def _raw_curves(records: list, mode: str) -> Dict[int, Dict[float, float]]:
    """
    For the given mode, return {n_options: {window: pct_before_gold}}.
    Only wrong predictions with index info are counted.
    """
    wrong = [
        r for r in records
        if r.get("mode") == mode
        and not r.get("correct", True)
        and r.get("predicted_index") is not None
        and r.get("gold_index") is not None
    ]
    groups: Dict[Tuple[int, float], list] = defaultdict(list)
    for r in wrong:
        groups[(int(r["n_options"]), r["window"])].append(r)

    all_n = sorted({k[0] for k in groups})
    result: Dict[int, Dict[float, float]] = {}
    for n in all_n:
        result[n] = {}
        for w in WINDOWS:
            recs = groups.get((n, w), [])
            if not recs:
                continue
            n_before = sum(1 for r in recs if r["predicted_index"] < r["gold_index"])
            result[n][w] = 100 * n_before / len(recs)
    return result


def _block_deviation_curves(records: list) -> Dict[int, Dict[float, float]]:
    """
    Block mode only. Deviation of BEFORE-the-block % from empirical null.

    BEFORE-the-block: predicted_index < block_start  (i.e. model escaped the
    original option group entirely toward earlier distractors).
    Null: block_start / (n_options − 1) × 100.
    Negative deviation = block anchoring (model stays in block more than random).
    """
    wrong = [
        r for r in records
        if r.get("mode") == "block"
        and not r.get("correct", True)
        and r.get("predicted_index") is not None
        and r.get("gold_index") is not None
    ]

    groups: Dict[Tuple[int, float], list] = defaultdict(list)
    for r in wrong:
        groups[(int(r["n_options"]), r["window"])].append(r)

    all_n = sorted({k[0] for k in groups})
    result: Dict[int, Dict[float, float]] = {}

    for n in all_n:
        result[n] = {}
        for w in WINDOWS:
            recs = groups.get((n, w), [])
            if not recs:
                continue
            b_start  = int(round(w * (n - 10)))
            emp_null = b_start / (n - 1) * 100
            n_before = sum(1 for r in recs if r["predicted_index"] < b_start)
            obs      = 100 * n_before / len(recs)
            result[n][w] = obs - emp_null

    return result


def _deviation_curves(records: list, mode: str) -> Dict[int, Dict[float, float]]:
    """
    For the given mode, return {n_options: {window: deviation_pp}}.

    deviation_pp = observed_pct_before_gold − empirical_null_pct
    empirical_null_pct = mean(gold_index / (n_options − 1) × 100) over wrong records
    """
    wrong = [
        r for r in records
        if r.get("mode") == mode
        and not r.get("correct", True)
        and r.get("predicted_index") is not None
        and r.get("gold_index") is not None
    ]

    # group by (n_options, window)
    groups: Dict[Tuple[int, float], list] = defaultdict(list)
    for r in wrong:
        groups[(int(r["n_options"]), r["window"])].append(r)

    all_n = sorted({k[0] for k in groups})
    result: Dict[int, Dict[float, float]] = {}

    for n in all_n:
        result[n] = {}
        for w in WINDOWS:
            recs = groups.get((n, w), [])
            if not recs:
                continue
            emp_null = sum(
                r["gold_index"] / (n - 1) * 100 for r in recs
            ) / len(recs)
            n_before = sum(1 for r in recs if r["predicted_index"] < r["gold_index"])
            obs = 100 * n_before / len(recs)
            result[n][w] = obs - emp_null

    return result


# ── drawing helpers ───────────────────────────────────────────────────────────

def _style_ax(ax, ylabel: str, title: str = "",
              ylim: Optional[Tuple[float, float]] = None,
              pct_fmt: str = "%+.0f%%",
              fontsize: float = 5.5) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Window position", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=2)
    ax.set_xticks(WINDOWS)
    ax.set_xticklabels([str(w) for w in WINDOWS], fontsize=fontsize - 0.5)
    ax.tick_params(labelsize=fontsize - 0.5)
    ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(pct_fmt))
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title, fontsize=fontsize + 0.5, fontweight="bold", pad=3)


def _draw_deviation_lines(ax, curves: Dict[int, Dict[float, float]],
                          all_n: List[int]) -> None:
    """Plot one line per N value onto ax."""
    for n in all_n:
        pts = curves.get(n, {})
        xs = [w for w in WINDOWS if w in pts]
        ys = [pts[w] for w in xs]
        if not xs:
            continue
        color  = N_PALETTE.get(n, "#333333")
        marker = N_MARKERS.get(n, "o")
        ax.plot(xs, ys,
                color=color, marker=marker, markersize=2.5,
                linewidth=0.9, zorder=3, label=f"N = {n:,}")


def _legend(fig, all_n: List[int], bbox: Tuple) -> None:
    handles = [
        mpatches.Patch(color=N_PALETTE.get(n, "#333333"), label=f"N = {n:,}")
        for n in all_n
    ]
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="lower center",
        ncol=len(all_n),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=bbox,
        title="N options",
        title_fontsize=9,
    )


# ── figure builders ───────────────────────────────────────────────────────────

def build_raw_figure(records: list) -> plt.Figure:
    """
    Supplementary figure: raw % wrong picks before gold (sparse mode),
    one line per N, with the diagonal positional null as a grey reference.
    """
    curves = _raw_curves(records, mode="sparse")
    all_n  = sorted(curves)

    FS = 5.5
    fig, ax = plt.subplots(1, 1, figsize=(2.1, 1.8))
    fig.subplots_adjust(left=0.20, right=0.97, top=0.97, bottom=0.22)

    ax.plot(WINDOWS, [w * 100 for w in WINDOWS],
            color="#AAAAAA", linewidth=0.9, linestyle="--",
            zorder=1, label="Unbiased model")
    ax.axhline(50, color="#EEEEEE", linewidth=0.6, linestyle=":", zorder=1)

    for n in all_n:
        pts = curves.get(n, {})
        xs = [w for w in WINDOWS if w in pts]
        ys = [pts[w] for w in xs]
        if not xs:
            continue
        ax.plot(xs, ys,
                color=N_PALETTE.get(n, "#333333"),
                marker=N_MARKERS.get(n, "o"),
                markersize=2.5, linewidth=0.9, zorder=3,
                label=f"N = {n:,}")

    _style_ax(ax,
              ylabel="Wrong picks before gold (%)",
              ylim=(0, 105),
              pct_fmt="%.0f%%",
              fontsize=FS)
    return fig


BLOCK_PALETTE = {
    "BEFORE": "#E6550D",   # orange-red
    "INSIDE": "#3182BD",   # blue
    "AFTER":  "#31A354",   # green
}
BLOCK_MARKERS = {"BEFORE": "o", "INSIDE": "s", "AFTER": "^"}


def _block_tripart_curves(records: list) -> Dict[int, Dict[str, Dict[float, float]]]:
    """
    For block mode wrong predictions, compute % BEFORE / INSIDE / AFTER per
    (n_options, window).

    block occupies [block_start, block_start + 10).
    block_start = int(round(window * (n_options - 10)))

    Returns {n_options: {"BEFORE": {window: pct}, "INSIDE": {...}, "AFTER": {...}}}.
    """
    wrong = [
        r for r in records
        if r.get("mode") == "block"
        and not r.get("correct", True)
        and r.get("predicted_index") is not None
        and r.get("gold_index") is not None
    ]

    groups: Dict[Tuple[int, float], list] = defaultdict(list)
    for r in wrong:
        groups[(int(r["n_options"]), r["window"])].append(r)

    all_n = sorted({k[0] for k in groups})
    result: Dict[int, Dict[str, Dict[float, float]]] = {
        n: {"BEFORE": {}, "INSIDE": {}, "AFTER": {}} for n in all_n
    }

    for (n, w), recs in groups.items():
        b_start = int(round(w * (n - 10)))
        b_end   = b_start + 10
        total   = len(recs)
        result[n]["BEFORE"][w] = 100 * sum(1 for r in recs if r["predicted_index"] < b_start) / total
        result[n]["INSIDE"][w] = 100 * sum(1 for r in recs if b_start <= r["predicted_index"] < b_end) / total
        result[n]["AFTER"][w]  = 100 * sum(1 for r in recs if r["predicted_index"] >= b_end) / total

    return result


def build_block_raw_figure(records: list) -> plt.Figure:
    """
    Block mode: one panel per N, three lines (BEFORE / INSIDE / AFTER).
    No title, no legend.
    """
    curves = _block_tripart_curves(records)
    all_n  = sorted(curves)

    fig, axes = plt.subplots(1, len(all_n), figsize=(4.5 * len(all_n), 4.0),
                             sharey=True)
    if len(all_n) == 1:
        axes = [axes]
    fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.12,
                        wspace=0.08)

    for ax, n in zip(axes, all_n):
        for cat in ("BEFORE", "INSIDE", "AFTER"):
            pts = curves[n][cat]
            xs  = [w for w in WINDOWS if w in pts]
            ys  = [pts[w] for w in xs]
            ax.plot(xs, ys,
                    color=BLOCK_PALETTE[cat],
                    marker=BLOCK_MARKERS[cat],
                    markersize=6, linewidth=2.0, zorder=3,
                    label=cat)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 105)
        ax.set_xticks(WINDOWS)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax.xaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
        ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Window position", fontsize=10)
        # N label as a text annotation instead of a title
        ax.text(0.5, 0.97, f"N = {n:,}", transform=ax.transAxes,
                ha="center", va="top", fontsize=10, fontweight="bold")

    axes[0].set_ylabel("Wrong picks (%)", fontsize=10)

    return fig


def build_block_raw_legend() -> plt.Figure:
    """Standalone legend figure for build_block_raw_figure."""
    handles = [
        plt.Line2D([0], [0], color=BLOCK_PALETTE[cat],
                   marker=BLOCK_MARKERS[cat], markersize=6, linewidth=2.0,
                   label=cat)
        for cat in ("BEFORE", "INSIDE", "AFTER")
    ]
    fig, ax = plt.subplots(figsize=(4.0, 0.5))
    ax.set_axis_off()
    fig.legend(handles, [h.get_label() for h in handles],
               loc="center", ncol=3, fontsize=9, frameon=False)
    return fig


def build_raw_legend(all_n: List[int]) -> plt.Figure:
    """Standalone legend figure for before_gold_raw."""
    handles = []
    handles.append(
        plt.Line2D([0], [0], color="#AAAAAA", linewidth=0.9, linestyle="--",
                   label="Unbiased model")
    )
    for n in all_n:
        handles.append(
            plt.Line2D([0], [0],
                       color=N_PALETTE.get(n, "#333333"),
                       marker=N_MARKERS.get(n, "o"),
                       markersize=6, linewidth=2.0,
                       label=f"N = {n:,}")
        )
    fig, ax = plt.subplots(figsize=(5.5, 0.5))
    ax.set_axis_off()
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="center", ncol=len(handles),
        fontsize=9, frameon=False,
    )
    return fig


def build_main_figure(records: list) -> plt.Figure:
    """
    Main paper figure: sparse mode only, deviation from null, one line per N.
    Single panel, no title, no legend.
    """
    curves = _deviation_curves(records, mode="sparse")
    all_n  = sorted(curves)

    FS = 5.5
    fig, ax = plt.subplots(1, 1, figsize=(2.1, 1.8))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.22)

    ax.axhline(0, color="#888888", linewidth=0.7, linestyle="--", zorder=1)
    _draw_deviation_lines(ax, curves, all_n)
    _style_ax(ax, ylabel="Deviation from unbiased model (%)", fontsize=FS)

    return fig


def build_main_legend(all_n: List[int]) -> plt.Figure:
    """Standalone legend figure for build_main_figure."""
    handles = [
        plt.Line2D([0], [0],
                   color=N_PALETTE.get(n, "#333333"),
                   marker=N_MARKERS.get(n, "o"),
                   markersize=6, linewidth=2.0,
                   label=f"N = {n:,}")
        for n in all_n
    ]
    fig, ax = plt.subplots(figsize=(3.5, 0.5))
    ax.set_axis_off()
    fig.legend(handles, [h.get_label() for h in handles],
               loc="center", ncol=len(handles), fontsize=9, frameon=False)
    return fig


def build_appendix_figure(records: list) -> plt.Figure:
    """
    Appendix figure: sparse (left) vs block (right), independent y-axes.

    Sparse — deviation of before-gold % from null   (positive = primacy bias).
    Block  — deviation of before-block % from null  (negative = block anchoring).
    """
    sparse_curves = _deviation_curves(records, mode="sparse")
    block_curves  = _block_deviation_curves(records)
    all_n = sorted(set(sparse_curves) | set(block_curves))

    def _ylim(curves, pad=5):
        vals = [v for pts in curves.values() for v in pts.values()]
        return (min(vals) - pad, max(vals) + pad)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.20,
                        wspace=0.30)

    for ax in axes:
        ax.axhline(0, color="#888888", linewidth=1.0, linestyle="--", zorder=1)

    _draw_deviation_lines(axes[0], sparse_curves, all_n)
    _style_ax(axes[0],
              ylabel="Deviation from unbiased model (pp)",
              title="Sparse mode  (before gold)",
              ylim=_ylim(sparse_curves))

    _draw_deviation_lines(axes[1], block_curves, all_n)
    _style_ax(axes[1],
              ylabel="Deviation from unbiased model (pp)",
              title="Block mode  (before block)",
              ylim=_ylim(block_curves))

    _legend(fig, all_n, bbox=(0.5, 0.01))

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    if args.completions_dir:
        records = load_records_from_completions(args.completions_dir)
    else:
        records = load_records_from_classified(args.classified_path)
        print(f"Loaded {len(records):,} high/medium-confidence records")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # print a quick summary table
    print("\nDeviation from empirical null (pp) — sparse mode:")
    sparse_curves = _deviation_curves(records, "sparse")
    block_curves  = _deviation_curves(records, "block")
    all_n = sorted(sparse_curves)
    header = f"  {'mode':<7}  {'N':<6}" + "".join(f"  w={w}" for w in WINDOWS)
    print(header)
    for mode_label, curves in [("sparse", sparse_curves), ("block", block_curves)]:
        for n in all_n:
            pts = curves.get(n, {})
            row = f"  {mode_label:<7}  {n:<6}" + "".join(
                f"  {pts[w]:+6.1f}" if w in pts else "     —  "
                for w in WINDOWS
            )
            print(row)

    # raw figure + standalone legend
    raw_curves = _raw_curves(records, mode="sparse")
    all_n_raw  = sorted(raw_curves)

    fig = build_raw_figure(records)
    p = out_dir / "before_gold_raw.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved → {p}")

    fig = build_raw_legend(all_n_raw)
    p = out_dir / "before_gold_raw_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    # main figure (deviation, sparse only) + standalone legend
    main_curves = _deviation_curves(records, mode="sparse")
    all_n_main  = sorted(main_curves)

    fig = build_main_figure(records)
    p = out_dir / "before_gold_main.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_main_legend(all_n_main)
    p = out_dir / "before_gold_main_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    # appendix figure (deviation, sparse vs block)
    fig = build_appendix_figure(records)
    p = out_dir / "before_gold_appendix.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    # block tripart figure + standalone legend
    fig = build_block_raw_figure(records)
    p = out_dir / "before_gold_block_tripart.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    fig = build_block_raw_legend()
    p = out_dir / "before_gold_block_tripart_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot positional bias (wrong picks before gold) figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Specify exactly one data source:\n"
            "  --completions-dir out/completions   (full dataset, recommended)\n"
            "  --classified-path path/to/classified_traces.jsonl  (strategy subset)\n"
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--completions-dir",
        help="Root completions directory (e.g. out/completions). "
             "All predictions.jsonl files are loaded.",
    )
    src.add_argument(
        "--classified-path",
        help="Path to classified_traces.jsonl (strategy-classified subset only).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    if args.output is None:
        if args.completions_dir:
            args.output = str(Path(args.completions_dir).parent / "figures" / "before_gold")
        else:
            args.output = str(Path(args.classified_path).parent / "figures" / "before_gold")

    main(args)
