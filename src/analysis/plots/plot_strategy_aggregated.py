"""
Aggregated strategy figure — two stacked panels (dependent / independent),
N=1000, sparse mode, averaged over all window positions.

Each panel shows one grouped bar cluster per model (S1, S2, S3, S4, Other).

Usage:
    python src/analysis/plot_strategy_aggregated.py \
        --classified-path out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────

TOP_STRATS_SPARSE = ["S1", "S2", "S3", "S4"]
TOP_STRATS_BLOCK  = ["S1", "S2", "S3", "S4", "S8"]
OTHER_LABEL       = "Other"

PALETTE = {
    "S1":    "#4C72B0",
    "S2":    "#DD8452",
    "S3":    "#55A868",
    "S4":    "#C44E52",
    "S8":    "#8172B2",
    "Other": "#CCCCCC",
}

# Only the four main models for the paper
MODELS_ORDERED = [
    "gpt-oss-120b",
    "gemini-3-flash-preview",
    "gemma-4-31B-it",
    "Qwen3.5-27B-FP8",
]

MODEL_LABEL = {
    "gpt-oss-120b":           "GPT-OSS-120B",
    "gemini-3-flash-preview":  "Gemini-Flash-3",
    "gemma-4-31B-it":         "Gemma-4-31B",
    "Qwen3.5-27B-FP8":        "Qwen-3.5-27B",
}

_CODE_RE = re.compile(r"^[Ss](\d+)")

def _norm(raw: str) -> str:
    m = _CODE_RE.match(raw.strip())
    return f"S{m.group(1)}" if m else raw


# ── data loading ──────────────────────────────────────────────────────────────

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


N_VALS = [100, 500, 1000]

def compute_pcts(records: list, dep_label: str, mode: str) -> dict:
    """Return {model_slug: {n: {strategy: pct}}} for all N, given mode and dep_label."""
    top_strats = TOP_STRATS_BLOCK if mode == "block" else TOP_STRATS_SPARSE
    all_strats = top_strats + [OTHER_LABEL]
    result = {}
    for model in MODELS_ORDERED:
        result[model] = {}
        for n_val in N_VALS:
            subset = [
                r for r in records
                if r.get("model_slug") == model
                and int(r.get("n_options", 0)) == n_val
                and r.get("mode") == mode
                and r.get("option_dep") == dep_label
            ]
            total = len(subset)
            if total == 0:
                result[model][n_val] = {s: 0.0 for s in all_strats}
                continue
            cnts = Counter(r["primary_strategy"] for r in subset)
            pcts = {s: 100 * cnts.get(s, 0) / total for s in top_strats}
            pcts[OTHER_LABEL] = max(0.0, 100 - sum(pcts.values()))
            result[model][n_val] = pcts
    return result


# ── drawing ───────────────────────────────────────────────────────────────────

BAR_W    = 0.22   # width of each stacked bar
N_GAP    = 0.03   # gap between N bars within a model group
MODEL_GAP = 0.35  # extra spacing between model groups (added to bar width)

def draw_panel(ax, dep_label: str, data: dict, mode: str = "sparse") -> None:
    all_strats = (TOP_STRATS_BLOCK if mode == "block" else TOP_STRATS_SPARSE) + [OTHER_LABEL]
    n_models = len(MODELS_ORDERED)
    n_n      = len(N_VALS)

    # x positions: 3 bars per model, small gap between models
    group_span = n_n * BAR_W + (n_n - 1) * N_GAP
    model_step = group_span + MODEL_GAP
    model_centers = np.arange(n_models) * model_step

    bar_offsets = np.arange(n_n) * (BAR_W + N_GAP) - group_span / 2 + BAR_W / 2

    for mi, model in enumerate(MODELS_ORDERED):
        for ni, n_val in enumerate(N_VALS):
            x   = model_centers[mi] + bar_offsets[ni]
            bot = 0.0
            for strat in all_strats:
                v = data[model][n_val][strat]
                ax.bar(x, v, bottom=bot, width=BAR_W * 0.92,
                       color=PALETTE[strat], zorder=3,
                       edgecolor="white", linewidth=0.3)
                # label inside segment if tall enough
                if v >= 8:
                    ax.text(x, bot + v / 2, f"{v:.0f}",
                            ha="center", va="center",
                            fontsize=6, color="white", fontweight="bold")
                bot += v

    # x-tick at model center, N labels below each bar
    ax.set_xticks(model_centers)
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS_ORDERED],
                       fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=18)

    # N labels under each bar
    for mi in range(n_models):
        for ni, n_val in enumerate(N_VALS):
            x = model_centers[mi] + bar_offsets[ni]
            ax.text(x, -4, f"N={n_val}", ha="center", va="top",
                    fontsize=7, color="#555555", rotation=0)

    ax.set_ylabel("Strategy share (%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    # light separator between model groups
    for mi in range(n_models - 1):
        sep = (model_centers[mi] + model_centers[mi + 1]) / 2
        ax.axvline(sep, color="#dddddd", linewidth=1.0, zorder=0)


    ax.set_xlim(model_centers[0] - group_span / 2 - 0.2,
                model_centers[-1] + group_span / 2 + 0.2)


# ── main ──────────────────────────────────────────────────────────────────────

RC = {
    "font.family": "serif",
    "font.size":   11,
    "figure.dpi":  150,
}

def save_panel(dep_label: str, data: dict, mode: str, out_dir: Path) -> None:
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.0))
    fig.subplots_adjust(bottom=0.18, top=0.97, left=0.09, right=0.97)
    draw_panel(ax, dep_label, data, mode=mode)
    slug = "dependent" if dep_label == "option_dependent" else "independent"
    out  = out_dir / f"strategy_aggregated_{mode}_{slug}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


def save_legend(mode: str, out_dir: Path) -> None:
    plt.rcParams.update(RC)
    strats  = (TOP_STRATS_BLOCK if mode == "block" else TOP_STRATS_SPARSE) + [OTHER_LABEL]
    handles = [mpatches.Patch(color=PALETTE[s], label=s) for s in strats]
    fig, ax = plt.subplots(figsize=(6.5 if mode == "block" else 5.5, 0.55))
    ax.set_axis_off()
    fig.legend(handles=handles, loc="center", ncol=len(strats),
               fontsize=11, frameon=False,
               title="Strategy", title_fontsize=11)
    out = out_dir / f"strategy_aggregated_{mode}_legend.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved → {out}")


def main(args):
    records = load_records(args.classified_path)
    print(f"Loaded {len(records)} high/medium-confidence records")

    out_dir = Path(args.classified_path).parent / "strategy2x2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("sparse", "block"):
        dep_data   = compute_pcts(records, "option_dependent",   mode)
        indep_data = compute_pcts(records, "option_independent", mode)
        save_panel("option_dependent",   dep_data,   mode, out_dir)
        save_panel("option_independent", indep_data, mode, out_dir)
        save_legend(mode, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified-path", required=True)
    args = parser.parse_args()
    main(args)
