"""
Build the final 1000-sample dataset from the combined categorized predictions.

Strategy:
  1. Include ALL 460 option_dependent samples (no dropping).
  2. Sample 540 option_independent samples such that the overall category
     distribution across the 1000 samples is as balanced as possible.

Category balancing algorithm:
  - Target: 1000 / num_categories samples per category.
  - For each category, the independent "budget" = max(0, target - dep_count).
  - If total budget > 540: scale proportionally (can't fully close all gaps).
  - If total budget < 540: fill all deficits first, then distribute the
    remaining slots to categories with the most available independent samples
    (further equalising totals).
  - Per-category budget is capped at the number of available independent
    samples in that category.
  - Rounding is done with Hamilton's largest-remainder method to hit
    exactly 540.
"""

import argparse
import random
from collections import defaultdict, Counter
from typing import Dict, List

from datasets import load_from_disk, Dataset


def largest_remainder(quotas: Dict[str, float], total: int) -> Dict[str, int]:
    """Allocate `total` integer slots proportionally via Hamilton's method."""
    floors = {k: int(v) for k, v in quotas.items()}
    remainders = {k: v - floors[k] for k, v in quotas.items()}
    allocated = sum(floors.values())
    extra = total - allocated
    for k in sorted(remainders, key=remainders.__getitem__, reverse=True)[:extra]:
        floors[k] += 1
    return floors


def compute_ind_budget(
    dep_counts: Dict[str, int],
    ind_available: Dict[str, int],
    n_independent: int,
) -> Dict[str, int]:
    categories = list(dep_counts.keys())
    n_cat = len(categories)
    target_per_cat = (sum(dep_counts.values()) + n_independent) / n_cat  # = 1000/4 = 250

    # Ideal independent budget per category to reach target
    ideal = {c: max(0.0, target_per_cat - dep_counts[c]) for c in categories}
    ideal_total = sum(ideal.values())

    if ideal_total == 0:
        # Edge case: dependent alone already meets or exceeds targets everywhere
        proportional = {c: float(ind_available[c]) for c in categories}
    elif ideal_total <= n_independent:
        # We can fill all deficits; distribute remaining slots by available independent
        remaining = n_independent - ideal_total
        extra_weight = {c: float(ind_available[c]) for c in categories}
        extra_total = sum(extra_weight.values())
        if extra_total > 0:
            proportional = {
                c: ideal[c] + remaining * extra_weight[c] / extra_total
                for c in categories
            }
        else:
            proportional = dict(ideal)
    else:
        # Budget is tight: scale ideal proportionally to sum to n_independent
        proportional = {c: ideal[c] * n_independent / ideal_total for c in categories}

    # Cap each category at its available independent count
    for c in categories:
        proportional[c] = min(proportional[c], float(ind_available[c]))

    # Re-scale after capping (iterate until stable)
    for _ in range(10):
        total_prop = sum(proportional.values())
        if abs(total_prop - n_independent) < 1e-6:
            break
        capped = {c for c in categories if proportional[c] >= ind_available[c] - 1e-9}
        free = {c for c in categories if c not in capped}
        if not free:
            break
        free_budget = n_independent - sum(proportional[c] for c in capped)
        free_total = sum(proportional[c] for c in free)
        if free_total > 0:
            for c in free:
                proportional[c] = proportional[c] * free_budget / free_total

    return largest_remainder(proportional, n_independent)


def build_final_dataset(combined_path: str, output_path: str, seed: int, total: int):
    print(f"Loading combined dataset from: {combined_path}")
    ds = load_from_disk(combined_path)

    # Split by label
    dependent = [r for r in ds if r["option_dependent_vs_independent"] == "option_dependent"]
    independent = [r for r in ds if r["option_dependent_vs_independent"] == "option_independent"]

    n_dep = len(dependent)
    n_ind_needed = total - n_dep

    print(f"\nDataset overview:")
    print(f"  option_dependent  : {n_dep}")
    print(f"  option_independent: {len(independent)}")
    print(f"  Total             : {len(ds)}")
    print(f"\nTarget: {total} samples ({n_dep} dependent + {n_ind_needed} independent)")

    # Group independent by category
    ind_by_cat: dict[str, list] = defaultdict(list)
    for r in independent:
        ind_by_cat[r["category"]].append(r)

    dep_counts = Counter(r["category"] for r in dependent)
    ind_available = {c: len(ind_by_cat[c]) for c in ind_by_cat}

    # Ensure all categories are represented even if dependent has 0 for some
    all_cats = sorted(set(list(dep_counts.keys()) + list(ind_available.keys())))
    for c in all_cats:
        dep_counts.setdefault(c, 0)
        ind_available.setdefault(c, 0)

    print(f"\n--- Category breakdown in dependent ({n_dep} samples) ---")
    target_per_cat = total / len(all_cats)
    for c in all_cats:
        print(f"  {c}: {dep_counts[c]}")

    ind_budget = compute_ind_budget(dep_counts, ind_available, n_ind_needed)

    print(f"\n--- Independent sampling budget (target/cat ≈ {target_per_cat:.0f}) ---")
    for c in all_cats:
        final_total = dep_counts[c] + ind_budget[c]
        print(f"  {c}: dep={dep_counts[c]} + ind={ind_budget[c]} = {final_total} "
              f"(avail independent: {ind_available[c]})")

    # Sample
    rng = random.Random(seed)
    sampled_ind = []
    for c in all_cats:
        pool = ind_by_cat[c]
        n = ind_budget[c]
        sampled_ind.extend(rng.sample(pool, n))

    final_rows = dependent + sampled_ind
    rng.shuffle(final_rows)

    final_ds = Dataset.from_list(final_rows)
    final_ds.save_to_disk(output_path)
    print(f"\nFinal dataset saved (HF format) to: {output_path}")

    jsonl_path = f"{output_path}.jsonl"
    final_ds.to_json(jsonl_path, orient="records", lines=True)
    print(f"Final dataset saved (JSONL format) to: {jsonl_path}")

    # Summary
    final_cats = Counter(r["category"] for r in final_rows)
    final_labels = Counter(r["option_dependent_vs_independent"] for r in final_rows)

    print(f"\n--- Final dataset: {len(final_rows)} samples ---")
    print(f"\nLabel distribution:")
    for label, count in final_labels.most_common():
        print(f"  {label}: {count} ({100 * count / len(final_rows):.1f}%)")

    print(f"\nCategory distribution:")
    for cat in all_cats:
        count = final_cats[cat]
        print(f"  {cat}: {count} ({100 * count / len(final_rows):.1f}%)")

    print(f"\nCategory × Label cross-tab:")
    print(f"  {'category':<20} {'dep':>6} {'ind':>6} {'total':>7}")
    print(f"  {'-'*42}")
    for cat in all_cats:
        rows_cat = [r for r in final_rows if r["category"] == cat]
        d = sum(1 for r in rows_cat if r["option_dependent_vs_independent"] == "option_dependent")
        i = sum(1 for r in rows_cat if r["option_dependent_vs_independent"] == "option_independent")
        print(f"  {cat:<20} {d:>6} {i:>6} {d+i:>7}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample final 1000-item dataset with balanced categories."
    )
    parser.add_argument(
        "--combined-path",
        type=str,
        default="mmlu_pro_downsampled_2858_categorized_combined",
        help="Path to the combined HF dataset produced by combine_predictions.py.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/bench/mmlu-pro-augmented-10",
        help="Output path for the final HF dataset.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=1000,
        help="Target total number of samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    build_final_dataset(
        combined_path=args.combined_path,
        output_path=args.output_path,
        seed=args.seed,
        total=args.total,
    )
