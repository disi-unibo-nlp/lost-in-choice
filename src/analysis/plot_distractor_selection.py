"""
Plot distractor selection rates across N options, block vs sparse, for all models.

For each (mode, N): compute mean and std of two metrics across the 5 window positions:
  - wrong_original_pct    = Any-Original (%) − Overall (%)
  - random_distractor_pct = Distractor (%)

Produces:
  out/distractor_plots/distractor_combined.pdf    2×2: (metric × mode), all models
  out/distractor_plots/distractor_model_<X>.pdf   1×2: (block | sparse), per model
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────

COMPLETIONS_DIR = Path("out/completions")
OUT_DIR = Path("out/distractor_plots")
SEED = 42
N_OPTIONS = [100, 500, 1000]
MODES = ["block", "sparse"]

MODELS = [
    "Qwen3.5-27B-FP8",
    "Qwen3.5-9B",
    "gemini-3-flash-preview",
    "gemma-4-31B-it",
    "gpt-oss-120b",
    "gpt-oss-20b",
]

MODEL_ABBREV = {
    "Qwen3.5-27B-FP8":        "Qwen-27B",
    "Qwen3.5-9B":             "Qwen-9B",
    "gemini-3-flash-preview":  "Gemini",
    "gemma-4-31B-it":         "Gemma",
    "gpt-oss-120b":           "OSS-120B",
    "gpt-oss-20b":            "OSS-20B",
}

MODEL_COLORS = {
    "Qwen3.5-27B-FP8":        "#4C72B0",
    "Qwen3.5-9B":             "#DD8452",
    "gemini-3-flash-preview":  "#55A868",
    "gemma-4-31B-it":         "#C44E52",
    "gpt-oss-120b":           "#8172B2",
    "gpt-oss-20b":            "#937860",
}

MODEL_MARKERS = {
    "Qwen3.5-27B-FP8":        "o",
    "Qwen3.5-9B":             "s",
    "gemini-3-flash-preview":  "^",
    "gemma-4-31B-it":         "D",
    "gpt-oss-120b":           "v",
    "gpt-oss-20b":            "P",
}

METRIC_COLORS = {
    "random_distractor": "#C44E52",
    "wrong_original":    "#4C72B0",
}
METRIC_LABELS = {
    "random_distractor": "Random distractor selected (%)",
    "wrong_original":    "Wrong original option selected (%)",
}
METRIC_LINESTYLE = {
    "random_distractor": "-",
    "wrong_original":    "--",
}


# ── data loading ────────────────────────────────────────────────────────────────

def load_stats() -> dict:
    """
    Returns stats[model][mode][n] = {
        "wrong_original":     (mean, std),
        "random_distractor":  (mean, std),
        "wrong_original_vals":    np.array of per-window values,
        "random_distractor_vals": np.array of per-window values,
    }
    """
    stats: dict = {}
    for model in MODELS:
        stats[model] = {}
        for mode in MODES:
            stats[model][mode] = {}
            for n in N_OPTIONS:
                csv = COMPLETIONS_DIR / model / f"summary_n{n}_seed{SEED}_{mode}.csv"
                if not csv.exists():
                    print(f"  [warn] missing: {csv}")
                    continue
                df = pd.read_csv(csv, index_col=0)
                wrong_orig = df["Any-Original (%)"] - df["Overall"]
                rand_dist  = df["Distractor (%)"]
                stats[model][mode][n] = {
                    "wrong_original":          (float(wrong_orig.mean()), float(wrong_orig.std())),
                    "random_distractor":       (float(rand_dist.mean()),  float(rand_dist.std())),
                    "wrong_original_vals":     wrong_orig.values,
                    "random_distractor_vals":  rand_dist.values,
                }
    return stats


# ── ax helpers ──────────────────────────────────────────────────────────────────

def _style_ax(ax, mode_title: str, xlabel: bool = True, ylabel: str = ""):
    ax.set_xscale("log")
    ax.set_xticks(N_OPTIONS)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_title(mode_title, fontsize=9, fontweight="bold")
    if xlabel:
        ax.set_xlabel("N options", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, color="#DDDDDD", linewidth=0.4)
    ax.set_axisbelow(True)


# ── combined figure (2 rows × 2 cols) ──────────────────────────────────────────

def build_combined(stats: dict) -> plt.Figure:
    """2×2: rows = (random_distractor, wrong_original), cols = (block, sparse), lines = models."""
    metrics = ["random_distractor", "wrong_original"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.subplots_adjust(left=0.09, right=0.82, top=0.92, bottom=0.09, hspace=0.35, wspace=0.22)

    for ri, metric in enumerate(metrics):
        for ci, mode in enumerate(MODES):
            ax = axes[ri][ci]
            for model in MODELS:
                mode_data = stats[model].get(mode, {})
                ns    = [n for n in N_OPTIONS if n in mode_data]
                means = [mode_data[n][metric][0] for n in ns]
                stds  = [mode_data[n][metric][1] for n in ns]
                if not ns:
                    continue
                color  = MODEL_COLORS[model]
                marker = MODEL_MARKERS[model]
                label  = MODEL_ABBREV[model]
                ax.plot(ns, means, color=color, marker=marker, markersize=5,
                        linewidth=1.6, label=label, zorder=3)
                lo = np.array(means) - np.array(stds)
                hi = np.array(means) + np.array(stds)
                ax.fill_between(ns, lo, hi, color=color, alpha=0.12, zorder=1)

            _style_ax(
                ax,
                mode_title=mode.capitalize(),
                xlabel=(ri == 1),
                ylabel=METRIC_LABELS[metric] if ci == 0 else "",
            )

    # Shared legend (right margin)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, frameon=False,
               bbox_to_anchor=(1.0, 0.5), title="Model", title_fontsize=8)

    fig.suptitle("Distractor selection vs N options — all models",
                 fontsize=10, fontweight="bold", y=0.97)
    return fig


# ── per-model figure (1×2) ─────────────────────────────────────────────────────

def build_per_model(stats: dict, model: str) -> plt.Figure:
    """
    1×2 (block | sparse).
    Each panel: solid line = random distractor, dashed = wrong original; ±1 std shading;
    individual window values as small semi-transparent dots.
    """
    abbrev = MODEL_ABBREV.get(model, model)
    metrics = ["random_distractor", "wrong_original"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=False)
    fig.subplots_adjust(left=0.1, right=0.82, top=0.84, bottom=0.16, wspace=0.25)

    for ci, mode in enumerate(MODES):
        ax = axes[ci]
        mode_data = stats[model].get(mode, {})

        for metric in metrics:
            ns = [n for n in N_OPTIONS if n in mode_data]
            if not ns:
                continue
            means = np.array([mode_data[n][metric][0] for n in ns])
            stds  = np.array([mode_data[n][metric][1] for n in ns])
            color = METRIC_COLORS[metric]
            ls    = METRIC_LINESTYLE[metric]
            label = METRIC_LABELS[metric].replace(" selected (%)", "")

            # Individual window dots with small jitter for clarity
            for i, n in enumerate(ns):
                vals = mode_data[n][f"{metric}_vals"]
                jitter = np.random.default_rng(seed=i).uniform(-n * 0.03, n * 0.03, size=len(vals))
                ax.scatter(n + jitter, vals, color=color, alpha=0.3, s=12, zorder=2, linewidths=0)

            ax.plot(ns, means, color=color, marker="o", markersize=5, linewidth=1.8,
                    ls=ls, label=label, zorder=4)
            ax.fill_between(ns, means - stds, means + stds, color=color, alpha=0.15, zorder=1)

        _style_ax(ax, mode_title=mode.capitalize(), xlabel=True,
                  ylabel="% of all predictions" if ci == 0 else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8.5, frameon=False,
               bbox_to_anchor=(1.0, 0.5))

    fig.suptitle(f"{abbrev} — distractor selection vs N options",
                 fontsize=10, fontweight="bold")
    return fig


# ── compact stacked-bar helpers ─────────────────────────────────────────────────

_WRONG_COLOR = "#4878D0"
_RAND_COLOR  = "#EE854A"
_N_ALPHA     = {100: 0.50, 500: 0.75, 1000: 1.00}
_BAR_W       = 0.20
_INTRA_GAP   = 0.04
_GROUP_GAP   = 0.55


def _compact_positions(intra_gap: float = _INTRA_GAP, group_gap: float = _GROUP_GAP):
    """Return (bar_x, model_cx): x-center for every (model, n) bar and each model group."""
    bar_x: dict = {}
    model_cx: list = []
    x = 0.0
    for model in MODELS:
        group_start = x
        for j, n in enumerate(N_OPTIONS):
            bar_x[(model, n)] = x + j * (_BAR_W + intra_gap)
        model_cx.append(group_start + (len(N_OPTIONS) - 1) * (_BAR_W + intra_gap) / 2)
        x = group_start + len(N_OPTIONS) * (_BAR_W + intra_gap) + group_gap
    return bar_x, model_cx


def _draw_compact_panel(ax, stats: dict, mode: str,
                        bar_x: dict, model_cx: list,
                        ylabel: bool = True, show_title: bool = True,
                        show_errorbars: bool = True,
                        fontsize_model: float = 7.5,
                        fontsize_ticks: float = 7.5,
                        fontsize_ylabel: float = 8.5,
                        model_yoffset: float = -0.14,
                        ytick_step: int = 10,
                        show_n_xticks: bool = True,
                        group_gap: float = _GROUP_GAP) -> None:
    """Draw one stacked-bar panel into ax. Error bars = ±1 std across window positions."""
    from matplotlib.transforms import blended_transform_factory

    for model in MODELS:
        mode_data = stats[model].get(mode, {})
        for n in N_OPTIONS:
            if n not in mode_data:
                continue
            xpos   = bar_x[(model, n)]
            alpha  = _N_ALPHA[n]
            wo = mode_data[n]["wrong_original"][0]
            rd = mode_data[n]["random_distractor"][0]

            ax.bar(xpos, wo, width=_BAR_W, color=_WRONG_COLOR, alpha=alpha,
                   zorder=3, linewidth=0)
            ax.bar(xpos, rd, width=_BAR_W, bottom=wo, color=_RAND_COLOR, alpha=alpha,
                   zorder=3, linewidth=0)
            if show_errorbars:
                total_vals = mode_data[n]["wrong_original_vals"] + mode_data[n]["random_distractor_vals"]
                total      = wo + rd
                total_std  = float(np.std(total_vals, ddof=1))
                yerr_lo    = total_std
                yerr_hi    = min(total_std, 100.0 - total)
                ax.errorbar(xpos, total, yerr=[[yerr_lo], [yerr_hi]], fmt="none",
                            ecolor="#555555", elinewidth=0.8, capsize=2, zorder=5)

    xtick_pos = [bar_x[(m, n)] for m in MODELS for n in N_OPTIONS]
    ax.set_xticks(xtick_pos)
    if show_n_xticks:
        ax.set_xticklabels([str(n) for _ in MODELS for n in N_OPTIONS],
                           fontsize=5.5, color="#555555")
        ax.tick_params(axis="x", length=2, pad=1)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for model, cx in zip(MODELS, model_cx):
        ax.text(cx, model_yoffset, MODEL_ABBREV.get(model, model),
                transform=trans, ha="center", va="top",
                fontsize=fontsize_model, fontweight="bold", clip_on=False)

    if show_title:
        ax.set_title(mode.capitalize(), fontsize=10, fontweight="bold", pad=6)
    if ylabel:
        ax.set_ylabel("% of all predictions", fontsize=fontsize_ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=fontsize_ticks)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ytick_step))
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-group_gap / 2,
                max(bar_x[(MODELS[-1], n)] for n in N_OPTIONS) + _BAR_W + group_gap / 2)


def _compact_legend_handles():
    from matplotlib.patches import Patch
    return [
        Patch(color=_WRONG_COLOR, alpha=1.0, label="Wrong original"),
        Patch(color=_RAND_COLOR,  alpha=1.0, label="Random distractor"),
        Patch(color="#888888",    alpha=0.50, label="N = 100"),
        Patch(color="#888888",    alpha=0.75, label="N = 500"),
        Patch(color="#888888",    alpha=1.00, label="N = 1000"),
    ]


def build_compact(stats: dict, show_errorbars: bool = True) -> plt.Figure:
    """1×2 stacked-bar figure (block | sparse), all models."""
    bar_x, model_cx = _compact_positions()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.22, wspace=0.06)

    for ci, mode in enumerate(MODES):
        _draw_compact_panel(axes[ci], stats, mode, bar_x, model_cx,
                            ylabel=(ci == 0), show_errorbars=show_errorbars)

    fig.legend(handles=_compact_legend_handles(), loc="lower center", ncol=5,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.2, handletextpad=0.4, columnspacing=1.0)
    title = ("Distractor selection breakdown — mean ± std over window positions"
             if show_errorbars else "Distractor selection breakdown — mean over window positions")
    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.97)
    return fig


_SINGLE_INTRA_GAP = 0.02
_SINGLE_GROUP_GAP = 0.24


def build_compact_single(stats: dict, mode: str, show_errorbars: bool = True,
                         ylim: tuple = None) -> plt.Figure:
    """Single stacked-bar panel for one mode — compact height, no title, no legend."""
    bar_x, model_cx = _compact_positions(intra_gap=_SINGLE_INTRA_GAP,
                                         group_gap=_SINGLE_GROUP_GAP)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.4))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.26)

    _draw_compact_panel(ax, stats, mode, bar_x, model_cx,
                        ylabel=True, show_title=False, show_errorbars=show_errorbars,
                        fontsize_model=6.5, fontsize_ticks=6.5,
                        fontsize_ylabel=7.5, model_yoffset=-0.10, ytick_step=10,
                        show_n_xticks=False, group_gap=_SINGLE_GROUP_GAP)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig


def build_compact_legend() -> plt.Figure:
    """Standalone horizontal legend figure."""
    fig, ax = plt.subplots(figsize=(5.5, 0.5))
    ax.set_axis_off()
    leg = ax.legend(handles=_compact_legend_handles(), loc="center", ncol=5,
                    fontsize=9, frameon=False,
                    handlelength=1.4, handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(pad=0.1)
    return fig


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = load_stats()

    fig = build_combined(stats)
    p = OUT_DIR / "distractor_combined.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    # Compute shared y-limits across both modes for consistent single-mode figures
    def _global_ylim(show_errorbars: bool) -> tuple:
        top_vals = []
        for mode in MODES:
            for model in MODELS:
                mode_data = stats[model].get(mode, {})
                for n in N_OPTIONS:
                    if n not in mode_data:
                        continue
                    wo = mode_data[n]["wrong_original"][0]
                    rd = mode_data[n]["random_distractor"][0]
                    top = wo + rd
                    if show_errorbars:
                        total_vals = (mode_data[n]["wrong_original_vals"]
                                      + mode_data[n]["random_distractor_vals"])
                        top += float(np.std(total_vals, ddof=1))
                    top_vals.append(top)
        return (0, max(top_vals) + 2)

    for show_std in [True, False]:
        suffix = "" if show_std else "_nostd"
        shared_ylim = _global_ylim(show_errorbars=show_std)

        fig = build_compact(stats, show_errorbars=show_std)
        p = OUT_DIR / f"distractor_compact{suffix}.pdf"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")

        for mode in MODES:
            fig = build_compact_single(stats, mode, show_errorbars=show_std,
                                       ylim=shared_ylim)
            p = OUT_DIR / f"distractor_compact_{mode}{suffix}.pdf"
            fig.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            print(f"Saved → {p}")

    fig = build_compact_legend()
    p = OUT_DIR / "distractor_compact_legend.pdf"
    fig.savefig(p, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {p}")

    for model in MODELS:
        fig = build_per_model(stats, model)
        slug = MODEL_ABBREV.get(model, model).replace(" ", "_").replace(".", "")
        p = OUT_DIR / f"distractor_model_{slug}.pdf"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved → {p}")


if __name__ == "__main__":
    main()
