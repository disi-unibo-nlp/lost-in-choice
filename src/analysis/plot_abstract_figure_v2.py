"""
Abstract figure v2 — warm editorial style inspired by reference screenshot.
Sparse placement, seed 42.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import to_rgb
import numpy as np
import os

# ── Data (sparse only, seed 42) ───────────────────────────────────────────────
WINDOWS = [0.00, 0.25, 0.50, 0.75, 1.00]

data = {
    "GPT-OSS-120B": {
        "baseline": 71.8,
        100:  [77.6, 68.1, 61.5, 58.9, 55.0],
        500:  [80.5, 57.8, 43.6, 39.4, 34.3],
        1000: [82.9, 48.8, 36.8, 27.6, 22.4],
    },
    "Gemini-Flash-3": {
        "baseline": 83.0,
        100:  [82.2, 81.7, 80.5, 78.5, 78.0],
        500:  [84.3, 76.9, 76.1, 74.6, 79.1],
        1000: [86.0, 76.5, 73.3, 73.4, 76.0],
    },
    "Gemma-4-31B": {
        "baseline": 79.3,
        100:  [77.1, 77.0, 75.5, 74.4, 74.0],
        500:  [82.3, 73.4, 65.7, 65.5, 69.2],
        1000: [83.9, 66.2, 66.2, 61.7, 65.8],
    },
    "Qwen-3.5-27B": {
        "baseline": 77.7,
        100:  [80.5, 76.3, 72.5, 71.7, 71.4],
        500:  [84.6, 72.0, 67.4, 64.3, 66.8],
        1000: [87.7, 70.9, 64.8, 63.3, 66.4],
    },
}

PANEL_ORDER = ["GPT-OSS-120B", "Gemini-Flash-3", "Gemma-4-31B", "Qwen-3.5-27B"]
N_VALS      = [100, 500, 1000]

# ── Warm editorial palette ─────────────────────────────────────────────────────
# light cream → amber → deep burgundy  (safe → danger)
COLORS  = ["#E8C547", "#D4693A", "#6B1F3E"]
MARKERS = ["o", "o", "o"]
DASHES  = [(4, 2), (4, 2), (4, 2)]   # all dashed, same rhythm
LWIDTH  = 2.4
MSIZE   = 10.5

AXES_BG    = "white"
BASELINE_C = "#555555"

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.facecolor":     AXES_BG,
    "figure.facecolor":   "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.frameon":     True,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         150,
})

fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), sharex=True)
fig.subplots_adjust(hspace=0.32, wspace=0.22,
                    bottom=0.14, top=0.97, left=0.09, right=0.96)

for ax, model_name in zip(axes.flat, PANEL_ORDER):
    md       = data[model_name]
    baseline = md["baseline"]

    # Warm background already set via rcParams; add subtle inner border tint
    for spine in ax.spines.values():
        spine.set_color("#aaaaaa")
        spine.set_linewidth(1.0)

    # Baseline dashed line
    ax.axhline(baseline, color=BASELINE_C, linewidth=1.8,
               linestyle=(0, (5, 4)), zorder=1, label="10-option baseline")

    # Lines N=100 / 500 / 1000
    for n_val, color, dash in zip(N_VALS, COLORS, DASHES):
        y = md[n_val]
        ax.plot(WINDOWS, y,
                color=color,
                linestyle=(0, dash),
                linewidth=LWIDTH,
                marker="o",
                markersize=MSIZE,
                markeredgecolor="white",
                markeredgewidth=1.4,
                zorder=4,
                label=f"$N={n_val}$")

    # Subtle fill between N=100 and N=1000
    y100  = np.array(md[100])
    y1000 = np.array(md[1000])
    ax.fill_between(WINDOWS, y100, y1000,
                    color="#cccccc", alpha=0.25, linewidth=0, zorder=2)

    # Model name — centered, large, bold
    ax.text(0.5, 0.97, model_name,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="top", ha="center", color="#2A0A14")

    # Axes ticks & labels
    ax.set_xticks(WINDOWS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    ax.tick_params(axis="both", labelsize=9, colors="#555555")
    for spine in ax.spines.values():
        spine.set_color("#C8A8A0")

    col_idx = list(axes.flat).index(ax) % 2
    if col_idx == 0:
        ax.set_ylabel("Overall accuracy (%)", fontsize=10, color="#444444")

    # Tight y-limits
    all_y = [v for n in N_VALS for v in md[n]]
    lo, hi = min(all_y), max(max(all_y), baseline)
    pad = (hi - lo) * 0.025
    ax.set_ylim(lo - pad, hi + pad)

    # Baseline annotation — plain bold text above the line
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.97, baseline,
            f"{baseline:.1f}%",
            transform=trans,
            fontsize=12, fontweight="bold", style="italic",
            va="bottom", ha="right",
            color="#333333")

# x-axis label on bottom row only
for ax in axes[1]:
    ax.set_xlabel("Gold position (window)", fontsize=10, color="#444444")

# Shared legend
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, 0.01),
           handlelength=2.0, columnspacing=1.6, borderpad=0.6,
           frameon=True, framealpha=0.95, edgecolor="#ddcccc")

os.makedirs("out/figures", exist_ok=True)
out = "out/figures/abstract_figure_v2.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200, pad_inches=0.15)
print(f"Saved → {out}")
