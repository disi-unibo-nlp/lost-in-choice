"""
Collect accuracy scores across seeds and models for N ∈ {100, 500, 1000}, then
print Markdown seed-robustness tables suitable for the paper appendix.

Usage:
    python src/analysis/seed_robustness_table.py [--n 100 500 1000]

For each (model, N, seed, window, mode) it picks the *most recent* timestamp
folder under out/completions/<model>/mmlu-pro-augmented-<N>__seed_<s>__window_<w>__<m>/
and reads metrics.json.  Missing combinations are shown as "—".

Mean and std are computed over all seeds with available results;
std is omitted when fewer than 2 seeds are available.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE = Path("out/completions")
ALL_N = [100, 500, 1000]
SEEDS = [42, 123, 456]
WINDOWS = [0.00, 0.25, 0.50, 0.75, 1.00]
MODES = ["block", "sparse"]

MODELS = {
    "gpt-oss-120b":    "GPT-oss-120b",
    "gemma-4-31B-it":  "Gemma-4-31B-it",
    "Qwen3.5-27B-FP8": "Qwen3.5-27B-FP8",
}

# N=100 uses 1000 questions; N=500 and N=1000 robustness runs use 100 questions.
N_QUESTIONS = {100: 1000, 500: 100, 1000: 100}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def most_recent_metrics(model: str, n: int, seed: int,
                         window: float, mode: str) -> Optional[dict]:
    tag = f"mmlu-pro-augmented-{n}__seed_{seed}__window_{window:.2f}__{mode}"
    folder = BASE / model / tag
    if not folder.exists():
        return None
    timestamps = sorted(folder.iterdir(), reverse=True)
    for ts in timestamps:
        p = ts / "metrics.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}"


def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mu = sum(values) / len(values)
    if len(values) < 2:
        return mu, None
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return mu, math.sqrt(variance)


# ---------------------------------------------------------------------------
# Collect data
# ---------------------------------------------------------------------------
def collect(ns: List[int]) -> dict:
    """data[model][n][seed][window][mode] = accuracy (float) or None"""
    data: dict = {}
    for model in MODELS:
        data[model] = {}
        for n in ns:
            data[model][n] = {}
            for seed in SEEDS:
                data[model][n][seed] = {}
                for w in WINDOWS:
                    data[model][n][seed][w] = {}
                    for mode in MODES:
                        m = most_recent_metrics(model, n, seed, w, mode)
                        data[model][n][seed][w][mode] = (
                            m["accuracy"] if m else None
                        )
    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
HEADER = (
    "| Window | Mode   |"
    " Seed 42 | Seed 123 | Seed 456 |"
    " Mean  |  Std  |"
)
SEP = (
    "|:------:|--------|"
    ":-------:|:--------:|:--------:|"
    ":-----:|:-----:|"
)


def model_block(data: dict, model: str, n: int) -> List[str]:
    lines = [HEADER, SEP]
    for mode in MODES:
        for w in WINDOWS:
            vals = [data[model][n][s][w][mode] for s in SEEDS]
            available = [v for v in vals if v is not None]
            mu, sigma = mean_std(available)

            cells = [fmt(v) for v in vals]
            mean_cell = fmt(mu) if mu is not None else "—"
            std_cell = (f"{sigma * 100:.1f}" if sigma is not None else "—")

            row = (
                f"| {w:.2f}   | {mode:<6} |"
                f" {cells[0]:>7} | {cells[1]:>8} | {cells[2]:>8} |"
                f" {mean_cell:>5} | {std_cell:>5} |"
            )
            lines.append(row)
        lines.append(
            f"| {'':6} | {'':6} | {'':7} | {'':8} | {'':8} | {'':5} | {'':5} |"
        )
    return lines[:-1]  # drop trailing blank separator


def macro_summary(data: dict, ns: List[int]) -> str:
    header = (
        "| Model | N | Questions | Seeds available |"
        " Mean σ (pp) | Max σ (pp) |"
    )
    sep = "|-------|:-:|:---------:|:---------------:|:-----------:|:----------:|"
    lines = ["**Macro-average σ across all 10 (window, mode) conditions**", "", header, sep]
    for model, label in MODELS.items():
        for n in ns:
            stds = []
            seeds_present = set()
            for mode in MODES:
                for w in WINDOWS:
                    vals = [data[model][n][s][w][mode] for s in SEEDS]
                    available = [v for v in vals if v is not None]
                    for i, v in enumerate(vals):
                        if v is not None:
                            seeds_present.add(SEEDS[i])
                    _, sigma = mean_std(available)
                    if sigma is not None:
                        stds.append(sigma)
            n_q = N_QUESTIONS[n]
            seeds_str = "+".join(str(s) for s in sorted(seeds_present)) if seeds_present else "—"
            if stds:
                mean_s = sum(stds) / len(stds) * 100
                max_s = max(stds) * 100
                lines.append(
                    f"| {label} | {n} | {n_q} | {seeds_str}"
                    f" | {mean_s:.1f} | {max_s:.1f} |"
                )
            else:
                lines.append(f"| {label} | {n} | {n_q} | {seeds_str} | — | — |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=ALL_N,
                        help="Pool sizes to include (default: 100 500 1000)")
    args = parser.parse_args()
    ns = args.n

    data = collect(ns)

    print("=" * 70)
    print("SEED ROBUSTNESS — accuracies in %")
    print("(most-recent run per condition)")
    print("=" * 70)
    print()

    for n in ns:
        n_q = N_QUESTIONS[n]
        print(f"## N = {n} options  ({n_q} questions per condition)")
        print()
        for model, label in MODELS.items():
            print(f"**{label}**")
            print()
            for line in model_block(data, model, n):
                print(line)
            print()

    print(macro_summary(data, ns))
    print()


if __name__ == "__main__":
    main()
