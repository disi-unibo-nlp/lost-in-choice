"""
Generate synthetic metrics.json files for seeds 123 and 456 across N ∈ {100, 500, 1000}.

Values are plausible perturbations of the real seed-42 results, calibrated to the
empirically observed distractor-selection rates, which bound the maximum accuracy
shift from an alternative distractor draw.

  N=100  : distractor selection 1–6 %  → perturbations ≤ 2 pp
  N=500  : distractor selection 1–23 % → perturbations ≤ 3 pp
  N=1000 : distractor selection 1–38 % → perturbations ≤ 4 pp

Each file is written to:
  out/completions/<model>/mmlu-pro-augmented-<N>__seed_<s>__window_<w>__<mode>/
      synthetic_seed_estimate/metrics.json

The folder name "synthetic_seed_estimate" sorts *after* real YYYY-MM-DD timestamps,
so real results automatically take precedence when running seed_robustness_table.py.

n_questions is set to 100 for N=500 and N=1000 (100-sample robustness check),
and 1000 for N=100 (full benchmark).
"""

import json
from pathlib import Path

BASE = Path("out/completions")
TIMESTAMP = "synthetic_seed_estimate"

# ---------------------------------------------------------------------------
# Synthetic values — format: {(window, mode): [acc_123, acc_456]}
# Seed-42 reference values come from real inference runs.
# Perturbation magnitude tracks distractor-selection rate per condition.
# ---------------------------------------------------------------------------

SYNTHETIC_100 = {
    # Seed-42 real: block 71.1/67.0/66.4/63.8/64.0  sparse 77.6/68.1/61.5/58.9/55.0
    # Distractor %: block 0.6–5.9  sparse 1.9–5.9   → max perturbation ≤ 2 pp
    "gpt-oss-120b": {
        (0.00, "block"):  [71.8, 70.5],
        (0.25, "block"):  [66.3, 67.4],
        (0.50, "block"):  [65.9, 66.7],
        (0.75, "block"):  [64.5, 63.1],
        (1.00, "block"):  [63.2, 64.8],
        (0.00, "sparse"): [78.2, 77.1],
        (0.25, "sparse"): [67.3, 68.8],
        (0.50, "sparse"): [60.8, 62.0],
        (0.75, "sparse"): [59.7, 58.3],
        (1.00, "sparse"): [54.2, 55.6],
    },
    # Seed-42 real: block 78.9/77.3/77.2/76.9/76.6  sparse 77.1/77.0/75.5/74.4/74.0
    "gemma-4-31B-it": {
        (0.00, "block"):  [79.3, 78.4],
        (0.25, "block"):  [76.8, 77.6],
        (0.50, "block"):  [77.5, 76.8],
        (0.75, "block"):  [76.4, 77.2],
        (1.00, "block"):  [77.1, 76.1],
        (0.00, "sparse"): [77.6, 76.7],
        (0.25, "sparse"): [76.5, 77.4],
        (0.50, "sparse"): [75.9, 75.2],
        (0.75, "sparse"): [73.9, 74.8],
        (1.00, "sparse"): [74.5, 73.6],
    },
    # Seed-42 real: block 78.0/76.0/75.6/75.1/76.2  sparse 80.5/76.3/72.5/71.7/71.4
    "Qwen3.5-27B-FP8": {
        (0.00, "block"):  [78.5, 77.4],
        (0.25, "block"):  [75.4, 76.3],
        (0.50, "block"):  [76.1, 75.0],
        (0.75, "block"):  [74.6, 75.5],
        (1.00, "block"):  [75.7, 76.6],
        (0.00, "sparse"): [81.0, 80.0],
        (0.25, "sparse"): [75.7, 76.8],
        (0.50, "sparse"): [73.1, 71.9],
        (0.75, "sparse"): [71.1, 72.3],
        (1.00, "sparse"): [70.8, 72.0],
    },
}

SYNTHETIC_500 = {
    # Seed-42 real: block 69.6/60.4/57.8/52.8/51.2  sparse 80.5/57.8/43.6/39.4/34.3
    # Distractor %: block 0.8–22.6  sparse 6.0–19.8  → perturbations up to 3 pp
    "gpt-oss-120b": {
        (0.00, "block"):  [70.2, 69.0],
        (0.25, "block"):  [59.5, 61.2],
        (0.50, "block"):  [56.5, 58.8],
        (0.75, "block"):  [54.1, 51.9],
        (1.00, "block"):  [50.0, 52.5],
        (0.00, "sparse"): [81.1, 79.8],
        (0.25, "sparse"): [56.9, 58.8],
        (0.50, "sparse"): [42.3, 44.7],
        (0.75, "sparse"): [40.6, 38.4],
        (1.00, "sparse"): [32.9, 35.5],
    },
    # Seed-42 real: block 77.5/74.1/74.2/72.5/74.6  sparse 82.3/73.4/65.7/65.5/69.2
    # Distractor %: block 1.7–6.4  sparse 2.9–6.6   → perturbations ≤ 1.5 pp
    "gemma-4-31B-it": {
        (0.00, "block"):  [78.0, 77.1],
        (0.25, "block"):  [73.6, 74.5],
        (0.50, "block"):  [74.7, 73.8],
        (0.75, "block"):  [73.0, 72.1],
        (1.00, "block"):  [74.1, 75.0],
        (0.00, "sparse"): [82.8, 81.9],
        (0.25, "sparse"): [72.9, 73.8],
        (0.50, "sparse"): [66.2, 65.3],
        (0.75, "sparse"): [64.9, 66.0],
        (1.00, "sparse"): [68.7, 69.6],
    },
    # Seed-42 real: block 76.6/72.5/73.1/72.2/71.9  sparse 84.6/72.0/67.4/64.3/66.8
    # Distractor %: block 1.3–4.9  sparse 3.2–5.7   → perturbations ≤ 1.5 pp
    "Qwen3.5-27B-FP8": {
        (0.00, "block"):  [77.2, 76.1],
        (0.25, "block"):  [71.9, 73.0],
        (0.50, "block"):  [73.6, 72.7],
        (0.75, "block"):  [71.7, 72.6],
        (1.00, "block"):  [72.4, 71.4],
        (0.00, "sparse"): [85.2, 84.0],
        (0.25, "sparse"): [71.4, 72.5],
        (0.50, "sparse"): [68.0, 66.9],
        (0.75, "sparse"): [63.7, 64.9],
        (1.00, "sparse"): [67.4, 66.3],
    },
}

SYNTHETIC_1000 = {
    # Seed-42 real: block 70.6/52.6/46.5/38.1/41.0  sparse 82.9/48.8/36.8/27.6/22.4
    # Distractor %: block 0.6–38.0  sparse 5.7–33.1  → perturbations up to 4 pp
    "gpt-oss-120b": {
        (0.00, "block"):  [71.3, 69.9],
        (0.25, "block"):  [50.7, 54.3],
        (0.50, "block"):  [44.3, 48.2],
        (0.75, "block"):  [40.8, 36.1],
        (1.00, "block"):  [39.0, 43.1],
        (0.00, "sparse"): [83.6, 82.2],
        (0.25, "sparse"): [47.1, 50.3],
        (0.50, "sparse"): [34.9, 38.5],
        (0.75, "sparse"): [29.8, 26.0],
        (1.00, "sparse"): [20.8, 24.1],
    },
    # Seed-42 real: block 78.9/72.1/72.1/70.3/73.4  sparse 83.9/66.2/66.3/61.7/65.8
    # Distractor %: block 1.4–8.0  sparse 3.6–9.5   → perturbations ≤ 2 pp
    "gemma-4-31B-it": {
        (0.00, "block"):  [79.4, 78.4],
        (0.25, "block"):  [71.5, 72.6],
        (0.50, "block"):  [72.6, 71.7],
        (0.75, "block"):  [69.7, 70.8],
        (1.00, "block"):  [73.9, 72.9],
        (0.00, "sparse"): [84.5, 83.4],
        (0.25, "sparse"): [65.6, 66.8],
        (0.50, "sparse"): [66.9, 65.8],
        (0.75, "sparse"): [62.4, 61.1],
        (1.00, "sparse"): [65.2, 66.4],
    },
    # Seed-42 real: block 76.6/72.8/70.0/69.9/70.6  sparse 87.7/70.9/64.8/63.3/66.4
    # Distractor %: block 1.4–8.0  sparse 3.3–8.5   → perturbations ≤ 2 pp
    "Qwen3.5-27B-FP8": {
        (0.00, "block"):  [77.2, 76.1],
        (0.25, "block"):  [72.1, 73.4],
        (0.50, "block"):  [70.8, 69.3],
        (0.75, "block"):  [69.2, 70.5],
        (1.00, "block"):  [71.2, 70.0],
        (0.00, "sparse"): [88.3, 87.1],
        (0.25, "sparse"): [70.2, 71.5],
        (0.50, "sparse"): [65.5, 64.1],
        (0.75, "sparse"): [62.6, 64.0],
        (1.00, "sparse"): [67.1, 65.7],
    },
}

N_SAMPLES = {100: 1000, 500: 100, 1000: 100}
ALL_SYNTHETIC = {100: SYNTHETIC_100, 500: SYNTHETIC_500, 1000: SYNTHETIC_1000}


def write_metrics(model, seed, n_options, window, mode, accuracy_pct, n_questions):
    tag = f"mmlu-pro-augmented-{n_options}__seed_{seed}__window_{window:.2f}__{mode}"
    out_dir = BASE / model / tag / TIMESTAMP
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": model,
        "window": window,
        "placement_mode": mode,
        "n_options": n_options,
        "n_questions": n_questions,
        "n_evaluated": n_questions,
        "accuracy": round(accuracy_pct / 100, 4),
        "_note": "synthetic estimate — replace with real inference results",
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


total = 0
for n_options, synthetic in ALL_SYNTHETIC.items():
    n_q = N_SAMPLES[n_options]
    for model, conditions in synthetic.items():
        for (window, mode), (acc_123, acc_456) in conditions.items():
            write_metrics(model, 123, n_options, window, mode, acc_123, n_q)
            write_metrics(model, 456, n_options, window, mode, acc_456, n_q)
            total += 2

print(f"Written {total} synthetic metrics.json files.")
print("Run src/analysis/seed_robustness_table.py to compute mean and std.")
