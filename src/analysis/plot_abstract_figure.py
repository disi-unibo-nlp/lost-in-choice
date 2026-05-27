"""
Abstract figure: 2x2 panel grid showing overall accuracy vs window position
for N=100, 500, 1000 (averaged across block and sparse modes), per model.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
from matplotlib.colors import to_rgb
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
# Extracted from summary_seed42.tex for each model.
# Format: {N: {window: avg(block, sparse)}}

WINDOWS = [0.00, 0.25, 0.50, 0.75, 1.00]

data = {
    "GPT-OSS-120B": {
        "baseline": 71.8,
        100: {
            "block":  [71.1, 67.0, 66.4, 63.8, 64.0],
            "sparse": [77.6, 68.1, 61.5, 58.9, 55.0],
        },
        500: {
            "block":  [69.6, 60.4, 57.8, 52.8, 51.2],
            "sparse": [80.5, 57.8, 43.6, 39.4, 34.3],
        },
        1000: {
            "block":  [70.6, 52.6, 46.5, 38.1, 41.0],
            "sparse": [82.9, 48.8, 36.8, 27.6, 22.4],
        },
    },
    "Gemini-Flash-3": {
        "baseline": 83.0,
        100: {
            "block":  [81.7, 81.2, 80.4, 81.2, 80.9],
            "sparse": [82.2, 81.7, 80.5, 78.5, 78.0],
        },
        500: {
            "block":  [81.2, 79.6, 79.3, 79.9, 80.1],
            "sparse": [84.3, 76.9, 76.1, 74.6, 79.1],
        },
        1000: {
            "block":  [80.5, 78.8, 78.7, 78.2, 79.6],
            "sparse": [86.0, 76.5, 73.3, 73.4, 76.0],
        },
    },
    "Gemma-4-31B": {
        "baseline": 79.3,
        100: {
            "block":  [78.9, 77.3, 77.2, 76.9, 76.6],
            "sparse": [77.1, 77.0, 75.5, 74.4, 74.0],
        },
        500: {
            "block":  [77.5, 74.1, 74.2, 72.5, 74.6],
            "sparse": [82.3, 73.4, 65.7, 65.5, 69.2],
        },
        1000: {
            "block":  [78.9, 72.1, 72.1, 70.3, 73.4],
            "sparse": [83.9, 66.2, 66.2, 61.7, 65.8],
        },
    },
    "Qwen-3.5-27B": {
        "baseline": 77.7,
        100: {
            "block":  [78.0, 76.0, 75.6, 75.1, 76.2],
            "sparse": [80.5, 76.3, 72.5, 71.7, 71.4],
        },
        500: {
            "block":  [76.6, 72.5, 73.1, 72.2, 71.9],
            "sparse": [84.6, 72.0, 67.4, 64.3, 66.8],
        },
        1000: {
            "block":  [76.6, 72.8, 70.0, 69.9, 70.6],
            "sparse": [87.7, 70.9, 64.8, 63.3, 66.4],
        },
    },
}

def get_sparse(model_data, n):
    return model_data[n]["sparse"]


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "grid.color": "#cccccc",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 150,
})

N_VALS   = [100, 500, 1000]
# ColorBrewer RdYlBu: safe blue → amber → danger red
COLORS   = ["#3182BD", "#F16913", "#99000D"]
MARKERS  = ["o", "s", "D"]
LWIDTHS  = [2.2, 2.2, 2.2]
MSIZES   = [7.5, 7.5, 7.0]
ALPHAS   = [1.0, 1.0, 1.0]

PANEL_ORDER = ["GPT-OSS-120B", "Gemini-Flash-3", "Gemma-4-31B", "Qwen-3.5-27B"]

fig, axes = plt.subplots(
    2, 2,
    figsize=(9, 7.5),
    sharex=True,
)
fig.subplots_adjust(hspace=0.32, wspace=0.22, bottom=0.14, top=0.97, left=0.09, right=0.96)

for ax, model_name in zip(axes.flat, PANEL_ORDER):
    model_data = data[model_name]
    baseline   = model_data["baseline"]

    # Baseline reference line (10-option standard benchmark)
    ax.axhline(
        baseline,
        color="#444444",
        linewidth=1.6,
        linestyle=(0, (4, 3)),   # custom dash: cleaner than ":"
        zorder=1,
        label="10-option baseline",
    )

    # Fill below baseline softly
    ax.axhspan(
        ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
        baseline,
        alpha=0.04,
        color="gray",
        zorder=0,
    )

    for n_val, color, marker, lw, ms, alpha in zip(
        N_VALS, COLORS, MARKERS, LWIDTHS, MSIZES, ALPHAS
    ):
        y = get_sparse(model_data, n_val)
        ax.plot(
            WINDOWS, y,
            color=color,
            marker=marker,
            markersize=ms,
            linewidth=lw,
            alpha=alpha,
            zorder=3,
            label=f"$N={n_val}$",
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    # Gradient shadow between N=100 (top) and N=1000 (bottom)
    y100  = np.array(get_sparse(model_data, 100))
    y1000 = np.array(get_sparse(model_data, 1000))
    c_top = np.array(to_rgb(COLORS[0]))   # blue  (N=100)
    c_bot = np.array(to_rgb(COLORS[2]))   # red   (N=1000)
    n_strips = 40
    for i in range(n_strips):
        t_lo  = i / n_strips
        t_hi  = (i + 1) / n_strips
        t_mid = (t_lo + t_hi) / 2
        yi_lo = y1000 + t_lo  * (y100 - y1000)
        yi_hi = y1000 + t_hi  * (y100 - y1000)
        c = tuple(c_bot * (1 - t_mid) + c_top * t_mid)
        ax.fill_between(WINDOWS, yi_lo, yi_hi,
                        color=c, alpha=0.13, linewidth=0, zorder=2)

    # Panel label — bold model name, centered at top
    ax.text(
        0.5, 0.97, model_name,
        transform=ax.transAxes,
        fontsize=14, fontweight="bold",
        va="top", ha="center",
        color="#111111",
    )

    # Axes decoration
    ax.set_xticks(WINDOWS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")
    ))
    # y-axis label only on left column
    col_idx = list(axes.flat).index(ax) % 2
    if col_idx == 0:
        ax.set_ylabel("Overall accuracy (%)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)

    # Tight y-limits — minimal padding so variation fills the panel
    all_y = [acc for n in N_VALS for acc in get_sparse(model_data, n)]
    lo, hi = min(all_y), max(max(all_y), baseline)
    pad = (hi - lo) * 0.025
    ax.set_ylim(lo - pad, hi + pad)

    # Baseline annotation: above the dashed line, near the right end
    # blended transform: x in axes [0,1], y in data coords
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.97, baseline,
        f"{baseline:.1f}%",
        transform=trans,
        fontsize=11, color="#333333",
        fontweight="bold", style="italic",
        va="bottom", ha="right",
    )

# x-axis label only on bottom row
for ax in axes[1]:
    ax.set_xlabel("Gold position (window)", fontsize=10)

# Single shared legend — placed in the reserved bottom margin
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=4,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.01),   # figure coords, inside bottom margin
    handlelength=2.0,
    columnspacing=1.6,
    borderpad=0.6,
    frameon=True,
    framealpha=0.95,
    edgecolor="#cccccc",
)


out_path = "out/figures/abstract_figure.pdf"
import os
os.makedirs("out/figures", exist_ok=True)
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200, pad_inches=0.15)
print(f"Saved → {out_path}")
