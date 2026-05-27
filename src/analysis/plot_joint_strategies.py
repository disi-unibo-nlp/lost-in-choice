"""
Generate joint strategy analysis figures for RQ1 + RQ3.

Four figures:
  A — Strategy distribution: dep × N stacked bars
  B — Accuracy surface: dep × strategy panels, lines per N, X=window
  C — Mode × dep stacked bars (highlights S8 spike)
  D — Token effort heatmap: strategy × model

Usage:
    python src/analysis/plot_joint_strategies.py \\
        --stats-path out/classification_batches/merged-qwen3.5-9b/strategy_stats.json \\
        --out-dir out/figures
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

MAIN_STRATEGIES = ["S1", "S2", "S3", "S4", "S8"]
ALL_DISPLAY = MAIN_STRATEGIES + ["Other"]

COLORS = {
    "S1":    "#4C72B0",
    "S2":    "#DD8452",
    "S3":    "#55A868",
    "S4":    "#C44E52",
    "S8":    "#8172B2",
    "Other": "#CCCCCC",
}

DEP_ORDER = ["option_independent", "option_dependent"]
DEP_LABELS = {
    "option_independent": "Option-independent",
    "option_dependent":   "Option-dependent",
}

N_STYLES = [
    ("100",  "o-",  "#1f77b4"),
    ("500",  "s--", "#ff7f0e"),
    ("1000", "^:",  "#2ca02c"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct_vec(dist: dict) -> dict:
    """Return {strategy: pct, Other: pct} for ALL_DISPLAY strategies."""
    named = {s: dist.get(s, {}).get("pct", 0.0) for s in MAIN_STRATEGIES}
    named["Other"] = max(0.0, 100.0 - sum(named.values()))
    return named


def _strategy_legend(ax=None, fig=None, loc="lower center", ncol=6, bbox=(0.5, -0.06)):
    handles = [mpatches.Patch(color=COLORS[s], label=s) for s in ALL_DISPLAY]
    target = fig if fig else ax
    target.legend(handles=handles, loc=loc, ncol=ncol,
                  bbox_to_anchor=bbox, fontsize=9)


# ---------------------------------------------------------------------------
# Figure A — dep × N stacked bars
# ---------------------------------------------------------------------------

def fig_A(s: dict, out_dir: Path) -> None:
    data   = s["strategy_x_dep_x_n"]
    n_opts = sorted({n for d in data.values() for n in d}, key=lambda x: int(x))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    fig.suptitle("Strategy distribution: option dependency × N options", fontsize=11)

    for ax, dep in zip(axes, DEP_ORDER):
        x       = np.arange(len(n_opts))
        bottoms = np.zeros(len(n_opts))
        for s_code in ALL_DISPLAY:
            heights = [_pct_vec(data[dep][n]["strategy_distribution"])[s_code] for n in n_opts]
            ax.bar(x, heights, bottom=bottoms, color=COLORS[s_code], label=s_code, width=0.55)
            bottoms += np.array(heights)
        ax.set_title(DEP_LABELS[dep], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in n_opts])
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        if dep == DEP_ORDER[0]:
            ax.set_ylabel("Strategy %")

    _strategy_legend(fig=fig, bbox=(0.5, -0.05))
    fig.tight_layout()
    path = out_dir / "fig_A_strategy_dep_x_n.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Figure B — accuracy surface: dep × strategy, lines per N, X=window
# ---------------------------------------------------------------------------

def fig_B(s: dict, out_dir: Path) -> None:
    acc     = s["accuracy_x_dep_x_n_x_window"]
    windows = sorted({w for d in acc.values() for n in d.values() for w in n}, key=float)
    xs      = [float(w) for w in windows]
    strategies = ["S1", "S2", "S3", "S4"]

    fig, axes = plt.subplots(
        len(DEP_ORDER), len(strategies),
        figsize=(13, 6), sharey=False, sharex=True,
    )
    fig.suptitle("Accuracy surface: strategy × dep × N × window", fontsize=11)

    for row_i, dep in enumerate(DEP_ORDER):
        for col_i, strat in enumerate(strategies):
            ax = axes[row_i][col_i]
            for n_opt, fmt, color in N_STYLES:
                ys_raw = [acc[dep].get(n_opt, {}).get(w, {}).get(strat) for w in windows]
                pairs  = [(x, y) for x, y in zip(xs, ys_raw) if y is not None]
                if not pairs:
                    continue
                px, py = zip(*pairs)
                ax.plot(px, [v * 100 for v in py], fmt, color=color,
                        label=f"N={n_opt}", markersize=5, linewidth=1.4)
            if row_i == 0:
                ax.set_title(strat, fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f"{DEP_LABELS[dep]}\nAccuracy %", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(w) for w in windows], fontsize=7)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.grid(True, alpha=0.25)
        axes[-1][col_i].set_xlabel("Window", fontsize=8)

    handles = [
        plt.Line2D([0], [0], marker=fmt[0], linestyle=fmt[1:], color=color, label=f"N={n}")
        for n, fmt, color in N_STYLES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig_B_accuracy_surface.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Figure C — mode × dep stacked bars (S8 spike callout)
# ---------------------------------------------------------------------------

def fig_C(s: dict, out_dir: Path) -> None:
    data    = s["strategy_x_dep_x_mode"]
    modes   = ["block", "sparse"]
    combos  = [(dep, mode) for dep in DEP_ORDER for mode in modes]
    labels  = [f"{DEP_LABELS[dep]}\n{mode}" for dep, mode in combos]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Strategy distribution: mode × option dependency", fontsize=11)

    x       = np.arange(len(combos))
    bottoms = np.zeros(len(combos))

    for s_code in ALL_DISPLAY:
        heights = [_pct_vec(data[dep][mode]["strategy_distribution"])[s_code]
                   for dep, mode in combos]
        bars = ax.bar(x, heights, bottom=bottoms, color=COLORS[s_code],
                      label=s_code, width=0.55)
        if s_code == "S8":
            for bar, h, b in zip(bars, heights, bottoms):
                if h >= 0.5:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, b + h / 2,
                        f"S8\n{h:.1f}%", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold",
                    )
        bottoms += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Strategy %")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    _strategy_legend(ax=ax, loc="upper right", ncol=2, bbox=(1, 1))
    fig.tight_layout()
    path = out_dir / "fig_C_mode_x_dep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Figure D — token effort heatmap: strategy × model
# ---------------------------------------------------------------------------

def fig_D(s: dict, out_dir: Path) -> None:
    tok_data = s.get("reasoning_tokens_x_strategy_x_model", {})
    if not tok_data:
        print("  Figure D skipped — no token data")
        return

    strategies = ["S1", "S2", "S3", "S4", "S8"]
    models     = sorted(tok_data.keys())

    matrix = np.full((len(strategies), len(models)), np.nan)
    for j, model in enumerate(models):
        for i, strat in enumerate(strategies):
            mean = tok_data[model].get(strat, {}).get("mean")
            if mean is not None:
                matrix[i, j] = mean

    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.4), 3.5))
    fig.suptitle("Mean completion tokens: strategy × model", fontsize=11)

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("-", "\n") for m in models], fontsize=8)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=9)

    vmax = np.nanmax(matrix)
    for i in range(len(strategies)):
        for j in range(len(models)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = "white" if v > vmax * 0.6 else "black"
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Mean tokens", shrink=0.8)
    fig.tight_layout()
    path = out_dir / "fig_D_tokens_strategy_x_model.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Figure E — full 4-axis heatmap: S1 proportion across N × window,
#             panels = dep × mode
# ---------------------------------------------------------------------------

def fig_E(s: dict, out_dir: Path) -> None:
    """
    2×2 stacked bar grid (dep × mode).
    Each panel: X = window, bars stacked by strategy (aggregated over N).
    Shows all strategies simultaneously across all 4 axes.
    """
    data4   = s["strategy_x_dep_x_n_x_window_x_mode"]
    modes   = ["block", "sparse"]
    n_opts  = sorted({n for dep in data4 for n in data4[dep]}, key=lambda x: int(x))
    windows = sorted({w for dep in data4 for n in data4[dep] for w in data4[dep][n]}, key=float)

    # bar geometry: 3 N-groups per window position
    # bar_width < inter-bar spacing so there is a visible gap between the three N bars
    n_groups      = len(n_opts)
    bar_width     = 0.17
    bar_spacing   = 0.25          # centre-to-centre distance within a group
    offsets       = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * bar_spacing
    group_centres = np.arange(len(windows))

    N_EDGE_COLORS = {n_opts[0]: "#333333", n_opts[1]: "#555555", n_opts[2]: "#777777"}
    N_HATCH      = {n_opts[0]: "",   n_opts[1]: "//",  n_opts[2]: "xx"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)
    fig.suptitle(
        "Strategy distribution across window × dep × mode  —  bars per N",
        fontsize=11,
    )

    for row_i, dep in enumerate(DEP_ORDER):
        for col_i, mode in enumerate(modes):
            ax = axes[row_i][col_i]

            for n_i, n_opt in enumerate(n_opts):
                xs      = group_centres + offsets[n_i]
                bottoms = np.zeros(len(windows))

                for s_code in ALL_DISPLAY:
                    heights = [
                        data4[dep].get(n_opt, {}).get(w, {}).get(mode, {})
                        .get("strategy_distribution", {}).get(s_code, {}).get("pct", 0.0)
                        for w in windows
                    ]
                    # Other = 100 - sum of named strategies
                    if s_code == "Other":
                        heights = [
                            max(0.0, 100.0 - sum(
                                data4[dep].get(n_opt, {}).get(w, {}).get(mode, {})
                                .get("strategy_distribution", {}).get(sc, {}).get("pct", 0.0)
                                for sc in MAIN_STRATEGIES
                            ))
                            for w in windows
                        ]
                    ax.bar(xs, heights, bottom=bottoms, color=COLORS[s_code],
                           width=bar_width, linewidth=0,
                           label=s_code if (row_i == 0 and col_i == 0 and n_i == 0) else "")
                    bottoms += np.array(heights)

            # Major ticks at group centres → window labels
            ax.set_xticks(group_centres)
            ax.set_xticklabels([str(w) for w in windows], fontsize=8)
            ax.tick_params(axis="x", which="major", length=6, pad=14)

            # Minor ticks at individual bar positions → N sub-labels
            bar_positions = [group_centres[g] + offsets[n_i]
                             for g in range(len(windows))
                             for n_i in range(len(n_opts))]
            _n_short      = {"100": "100", "500": "500", "1000": "1k"}
            n_sublabels   = [_n_short.get(n_opts[n_i], n_opts[n_i])
                             for g in range(len(windows))
                             for n_i in range(len(n_opts))]
            ax.set_xticks(bar_positions, minor=True)
            ax.set_xticklabels(n_sublabels, minor=True, fontsize=5.5, color="#555555")
            ax.tick_params(axis="x", which="minor", length=0, pad=2)

            ax.set_xlim(-0.65, len(windows) - 0.25)
            ax.set_ylim(0, 100)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            if row_i == 0:
                ax.set_title(mode, fontsize=10, pad=8)
            if col_i == 0:
                ax.set_ylabel(f"{DEP_LABELS[dep]}\nStrategy %", fontsize=9)
            if row_i == len(DEP_ORDER) - 1:
                ax.set_xlabel("Window position", fontsize=9, labelpad=18)

    _strategy_legend(fig=fig, bbox=(0.5, 0.0), ncol=6)
    fig.subplots_adjust(top=0.93, bottom=0.13, hspace=0.35, wspace=0.12)
    path = out_dir / "fig_E_strategy_4axis.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    stats_path = Path(args.stats_path)
    if not stats_path.exists():
        raise SystemExit(f"Not found: {stats_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(stats_path) as f:
        full = json.load(f)
    s = full["main"]

    print(f"Generating figures → {out_dir}/")
    fig_A(s, out_dir)
    fig_B(s, out_dir)
    fig_C(s, out_dir)
    fig_D(s, out_dir)
    fig_E(s, out_dir)

    if args.classified_path:
        traces = _load_traces(Path(args.classified_path))
        fig_F(traces, out_dir)
    else:
        print("  Figure F skipped — pass --classified-path to generate it.")

    print("Done.")

# ---------------------------------------------------------------------------
# Figure F — 5-axis heatmap: model × N rows, window cols, dep × mode panels
# ---------------------------------------------------------------------------

_CODE_RE = re.compile(r"^[Ss](\d+)")

MODEL_ORDER = [
    "gpt-oss-20b", "gpt-oss-120b", "gemini-3-flash-preview",
    "gemma-4-31B-it", "Qwen3.5-9B", "Qwen3.5-27B-FP8",
]
MODEL_SHORT = {
    "gpt-oss-20b":            "GPT-20B",
    "gpt-oss-120b":           "GPT-120B",
    "gemini-3-flash-preview": "Gemini-Flash",
    "gemma-4-31B-it":         "Gemma-31B",
    "Qwen3.5-9B":             "Qwen-9B",
    "Qwen3.5-27B-FP8":        "Qwen-27B",
}


def _load_traces(path: Path) -> list:
    _norm = lambda c: f"S{m.group(1)}" if (m := _CODE_RE.match(c.strip())) else c.strip()
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("confidence", "low") not in ("high", "medium"):
                continue
            r["primary_strategy"] = _norm(r.get("primary_strategy", ""))
            records.append(r)
    print(f"  Loaded {len(records)} trace records for Figure F")
    return records


def _dist_from_traces(records: list) -> dict:
    total = len(records)
    if not total:
        return {}
    counts: dict = defaultdict(int)
    for r in records:
        counts[r["primary_strategy"]] += 1
    return {s: 100 * counts[s] / total for s in MAIN_STRATEGIES}


def fig_F(traces: list, out_dir: Path) -> None:
    """
    5-axis strategy heatmap.
    Panels : 2×2 (dep × mode), each with its own colour range (fix 3).
    Rows   : model × N — model name annotated once per group (fix 1).
    Cols   : window (5).
    Cell   : S1 % only, one number (fix 2). S2 shown in corner only if ≥ 15 %.
    Groups : separated by thick black lines (fix 4).
    """
    from matplotlib.gridspec import GridSpec

    modes   = ["block", "sparse"]
    n_opts  = sorted({str(r["n_options"]) for r in traces}, key=lambda x: int(x))
    windows = sorted({str(r["window"]) for r in traces}, key=float)
    models  = [m for m in MODEL_ORDER if any(r.get("model_slug") == m for r in traces)]

    n_rows = len(models) * len(n_opts)   # 18
    n_cols = len(windows)                # 5

    # Pre-group once
    grouped: dict = defaultdict(list)
    for r in traces:
        grouped[(
            r.get("option_dep", ""), r.get("mode", ""),
            r.get("model_slug", ""), str(r.get("n_options", "")),
            str(r.get("window", "")),
        )].append(r)

    # Build matrices
    matrices: dict = {}
    for dep in DEP_ORDER:
        for mode in modes:
            s1_mat = np.full((n_rows, n_cols), np.nan)
            s2_mat = np.full((n_rows, n_cols), np.nan)
            for r_i, (model, n_opt) in enumerate(
                (m, n) for m in models for n in n_opts
            ):
                for c_i, window in enumerate(windows):
                    d = _dist_from_traces(grouped[(dep, mode, model, n_opt, window)])
                    s1_mat[r_i, c_i] = d.get("S1", 0.0)
                    s2_mat[r_i, c_i] = d.get("S2", 0.0)
            matrices[(dep, mode)] = (s1_mat, s2_mat)

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(
        "S1 proportion (%, color) — model × N × window × dep × mode\n"
        "S2% shown in cell corner when ≥15%",
        fontsize=11, y=1.01,
    )
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.035],
                  wspace=0.22, hspace=0.38)

    last_im = {dep: None for dep in DEP_ORDER}

    for row_i, dep in enumerate(DEP_ORDER):
        # Fix 3: per-dep colour range
        dep_s1_min = min(np.nanmin(matrices[(dep, m)][0]) for m in modes)
        dep_s1_max = max(np.nanmax(matrices[(dep, m)][0]) for m in modes)

        for col_i, mode in enumerate(modes):
            ax = fig.add_subplot(gs[row_i, col_i])
            s1_mat, s2_mat = matrices[(dep, mode)]

            im = ax.imshow(s1_mat, aspect="auto", cmap="Blues",
                           vmin=dep_s1_min, vmax=dep_s1_max)
            last_im[dep] = im

            mid  = (dep_s1_min + dep_s1_max) * 0.65
            for ri in range(n_rows):
                for ci in range(n_cols):
                    s1v = s1_mat[ri, ci]
                    s2v = s2_mat[ri, ci]
                    fg  = "white" if s1v > mid else "black"
                    # Fix 2: single bold S1 number, centred
                    ax.text(ci, ri, f"{s1v:.0f}",
                            ha="center", va="center",
                            fontsize=7, color=fg, fontweight="bold")
                    # S2 only in top-right corner when notable
                    if s2v >= 15.0:
                        ax.text(ci + 0.42, ri - 0.35, f"{s2v:.0f}",
                                ha="right", va="top",
                                fontsize=5.5,
                                color="white" if fg == "white" else "#555555")

            # Fix 4: thick black separators between model groups
            for sep in range(len(n_opts), n_rows, len(n_opts)):
                ax.axhline(sep - 0.5, color="black", linewidth=2.0, alpha=0.7)

            # Fix 1: N as ytick, model name annotated once per group
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(
                [n for _ in models for n in n_opts], fontsize=7
            )
            if col_i == 0:
                for m_i, model in enumerate(models):
                    mid_row = m_i * len(n_opts) + (len(n_opts) - 1) / 2
                    ax.annotate(
                        MODEL_SHORT[model],
                        xy=(-0.5, mid_row), xycoords="data",
                        ha="right", va="center",
                        fontsize=7, fontweight="bold",
                        annotation_clip=False,
                    )

            ax.set_xticks(range(n_cols))
            ax.set_xticklabels([str(w) for w in windows], fontsize=8)
            ax.set_title(mode, fontsize=10, pad=6)
            if col_i == 0:
                ax.set_ylabel(DEP_LABELS[dep], fontsize=9, labelpad=65)
            if row_i == len(DEP_ORDER) - 1:
                ax.set_xlabel("Window", fontsize=8)

    # One colorbar per dep row, placed in the shared third column
    for row_i, dep in enumerate(DEP_ORDER):
        cbar_ax = fig.add_subplot(gs[row_i, 2])
        fig.colorbar(last_im[dep], cax=cbar_ax, label="S1 %")

    path = out_dir / "fig_F_model_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate joint strategy figures (RQ1 + RQ3)."
    )
    parser.add_argument(
        "--stats-path",
        default="out/classification_batches/merged-qwen3.5-9b/strategy_stats.json",
        help="Path to strategy_stats.json (default: merged-qwen3.5-9b).",
    )
    parser.add_argument(
        "--classified-path",
        default=None,
        help="Path to classified_traces.jsonl — required for Figure F (model breakdown).",
    )
    parser.add_argument(
        "--out-dir",
        default="out/figures",
        help="Output directory for PNG files (default: out/figures).",
    )
    args = parser.parse_args()
    main(args)
