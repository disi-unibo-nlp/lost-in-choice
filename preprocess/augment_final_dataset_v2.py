"""
Augment the final 1000-question dataset to N options per question.  (v2)

Produces 10 variants (5 window positions × 2 placement modes):

  Window positions (block anchor):
    i=0.00  →  block starts at index 0
    i=0.25  →  block starts at round(0.25 × (N - 10))
    i=0.50  →  block starts at round(0.50 × (N - 10))
    i=0.75  →  block starts at round(0.75 × (N - 10))
    i=1.00  →  block starts at N - 10  (block occupies the last 10 slots)

  The gold answer inherits its natural position *within* the block, so its
  absolute index is block_start + answer_index  (±0–9 spread per question).

  Placement modes:
    block  — the original 10 options stay contiguous at block_start.
    sparse — the gold answer is placed at the SAME absolute index as in block
             mode; the other 9 originals are spread one-per-segment across the
             remaining 9 equal segments of the option list.

  This design means block vs. sparse differ only in the arrangement of the
  non-gold originals.  Any accuracy difference is attributable purely to option
  clustering, not to a different gold position.

Distractor pool: all option texts (correct AND incorrect) from ALL questions
in the SAME CATEGORY in the source dataset (2858 questions, including the
bench 1000).  At sample time the current question's own 10 options are
excluded.  Effective pool sizes comfortably exceed 990 slots for every
category (computer science: ~2327, health: ~4761, law: ~6321, math: ~7073),
so the with-replacement fallback is never triggered in practice.

Output layout:
  <output_root>/seed_<N>/window_<pos>/block/       — HF dataset
  <output_root>/seed_<N>/window_<pos>/block.jsonl  — JSONL flat file
  <output_root>/seed_<N>/window_<pos>/sparse/
  <output_root>/seed_<N>/window_<pos>/sparse.jsonl
"""

import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset, load_from_disk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODES = ["block", "sparse"]
N_ORIG = 10       # original options per question
N_TARGET = 1000   # default augmented options per question


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def block_start(window: float, n: int, b: int = N_ORIG) -> int:
    """First index of the original-options block in the augmented list.

    Guaranteed to be in [0, n - b], so the full block always fits without
    clamping regardless of n or b.
    """
    return int(round(window * (n - b)))


def sample_distractors(
    pool: List[str],
    exclude: List[str],
    needed: int,
    rng: random.Random,
) -> List[str]:
    """Sample `needed` distractors from `pool`, avoiding strings in `exclude`."""
    exclude_set = set(exclude)
    candidates = [t for t in pool if t not in exclude_set]
    if len(candidates) >= needed:
        return rng.sample(candidates, needed)
    # Not enough unique candidates — fill with replacement
    result = list(candidates)
    result += rng.choices(candidates if candidates else pool, k=needed - len(result))
    return result


def augment_block(
    original_options: List[str],
    correct_text: str,
    distractors: List[str],
    window: float,
    rng: random.Random,
    n: int = N_TARGET,
) -> Tuple[List[str], int, int]:
    """
    Insert the original 10 options as a contiguous block at block_start(window, n).
    Original option order is preserved.

    Returns:
        new_options       — full list of n options
        new_answer_index  — absolute index of correct_text in new_options
        b_start           — where the block was inserted
    """
    b_start = block_start(window, n)
    others = list(distractors)
    rng.shuffle(others)
    block = list(original_options)   # order untouched
    result = others[:b_start] + block + others[b_start:]
    gold_in_block = block.index(correct_text)
    new_answer_index = b_start + gold_in_block
    return result, new_answer_index, b_start


def augment_sparse(
    original_options: List[str],
    correct_text: str,
    distractors: List[str],
    window: float,
    rng: random.Random,
    n: int = N_TARGET,
) -> Tuple[List[str], int, int]:
    """
    Spread the original 10 options one-per-segment across n slots.

    The gold answer lands at block_start(window, n) + answer_index, which is
    IDENTICAL to the gold position in augment_block for the same question.
    The other 9 originals each go to a random free slot in their own segment.

    Segments: 10 equal partitions of [0, n), each of size n // N_ORIG.

    Returns:
        new_options       — full list of n options
        new_answer_index  — absolute index of correct_text (== block mode's index)
        b_start           — same value as block_start(window, n)
    """
    assert n % N_ORIG == 0, f"n={n} must be divisible by N_ORIG={N_ORIG}"
    b_start = block_start(window, n)
    gold_in_block = list(original_options).index(correct_text)
    gold_idx = b_start + gold_in_block   # identical to block-mode gold index

    seg_size = n // N_ORIG
    gold_seg = gold_idx // seg_size

    other_originals = [o for o in original_options if o != correct_text]
    rng.shuffle(other_originals)
    other_segs = [s for s in range(N_ORIG) if s != gold_seg]

    result: List[str] = [None] * n  # type: ignore[list-item]
    occupied: set = set()

    result[gold_idx] = correct_text
    occupied.add(gold_idx)

    for seg, opt in zip(other_segs, other_originals):
        seg_start = seg * seg_size
        free = [i for i in range(seg_start, seg_start + seg_size) if i not in occupied]
        chosen = rng.choice(free)
        result[chosen] = opt
        occupied.add(chosen)

    free_slots = [i for i in range(n) if result[i] is None]
    dist_shuffled = list(distractors)
    rng.shuffle(dist_shuffled)
    for slot, d in zip(free_slots, dist_shuffled):
        result[slot] = d

    return result, gold_idx, b_start


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------

def build_category_pools(source_dataset) -> Dict[str, List[str]]:
    """Per-category distractor pool built from all questions in source_dataset."""
    pools: Dict[str, set] = defaultdict(set)
    for ex in source_dataset:
        for opt in ex["options"]:
            pools[ex["category"]].add(opt)

    unique_pools = {cat: sorted(texts) for cat, texts in pools.items()}

    print("\n--- Distractor pool sizes (all questions, same category) ---")
    for cat in sorted(unique_pools):
        print(f"  {cat}: {len(unique_pools[cat])} unique option texts")
    print()
    return unique_pools


def augment_dataset(
    dataset,
    category_pools: Dict[str, List[str]],
    window: float,
    mode: str,
    seed: int = 42,
    n: int = N_TARGET,
) -> Dataset:
    rng = random.Random(seed)
    records = []
    for ex in dataset:
        ans_idx = ex["answer_index"]
        correct_text = ex["options"][ans_idx]
        original_options = list(ex["options"])
        cat = ex["category"]

        pool = category_pools.get(cat, [])
        distractors = sample_distractors(
            pool,
            exclude=original_options,
            needed=n - N_ORIG,
            rng=rng,
        )

        if mode == "block":
            new_options, new_answer_index, b_start_val = augment_block(
                original_options, correct_text, distractors, window, rng, n
            )
        elif mode == "sparse":
            new_options, new_answer_index, b_start_val = augment_sparse(
                original_options, correct_text, distractors, window, rng, n
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        assert new_options[new_answer_index] == correct_text, (
            f"Answer mismatch for qid={ex.get('question_id','?')}: "
            f"idx={new_answer_index}, got={new_options[new_answer_index]!r}, "
            f"expected={correct_text!r}"
        )
        assert len(new_options) == n, f"Expected {n} options, got {len(new_options)}"
        assert new_answer_index == b_start_val + ans_idx, (
            f"Gold index formula violated: {new_answer_index} != {b_start_val} + {ans_idx}"
        )

        record = dict(ex)
        record["new_options"] = new_options
        record["new_answer_index"] = new_answer_index
        record["block_start_index"] = b_start_val
        record["window"] = window
        record["placement_mode"] = mode
        records.append(record)

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    if args.output_root is None:
        args.output_root = f"data/bench/mmlu-pro-augmented-{args.n_options}"

    print(f"Loading final dataset from: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    print(f"  {len(ds)} questions loaded")

    print(f"\nLoading source dataset from: {args.source_dataset_path}")
    source_ds = load_from_disk(args.source_dataset_path)
    print(f"  {len(source_ds)} questions in source")

    category_pools = build_category_pools(source_ds)

    n_variants = len(args.seeds) * len(WINDOW_POSITIONS) * len(MODES)
    print(f"Generating {n_variants} variants "
          f"({len(args.seeds)} seeds × {len(WINDOW_POSITIONS)} windows × {len(MODES)} modes)\n")

    for seed in args.seeds:
        seed_dir = os.path.join(args.output_root, f"seed_{seed}")
        print(f"\n{'='*52}")
        print(f"  Seed {seed}  →  {seed_dir}")
        print(f"{'='*52}")

        for window in WINDOW_POSITIONS:
            b_start_val = block_start(window, args.n_options)
            label = f"window_{window:.2f}"
            print(f"\n  [{label}]  block_start = {b_start_val}  "
                  f"(gold range [{b_start_val}, {b_start_val + N_ORIG - 1}])")

            for mode in MODES:
                aug_ds = augment_dataset(
                    ds,
                    category_pools=category_pools,
                    window=window,
                    mode=mode,
                    seed=seed,
                    n=args.n_options,
                )

                indices = [r["new_answer_index"] for r in aug_ds]
                print(f"    [{mode}]  gold index — min={min(indices)}  "
                      f"max={max(indices)}  mean={sum(indices)/len(indices):.1f}")

                out_dir = os.path.join(seed_dir, label, mode)
                os.makedirs(out_dir, exist_ok=True)
                aug_ds.save_to_disk(out_dir)

                jsonl_path = os.path.join(seed_dir, label, f"{mode}.jsonl")
                aug_ds.to_json(jsonl_path, orient="records", lines=True)
                print(f"    Saved → {out_dir}/")

    print(f"\nAll {n_variants} variants saved under: {args.output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment the final dataset to N options (v2: block-anchor positioning)."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/bench/mmlu-pro-augmented-10",
        help="Path to the 10-option bench dataset.",
    )
    parser.add_argument(
        "--source-dataset-path",
        type=str,
        default="mmlu_pro_downsampled_2858",
        help="Path to the full downsampled source dataset for distractor pools.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root directory for output. Defaults to data/bench/mmlu-pro-augmented-<n-options>.",
    )
    parser.add_argument(
        "--n-options",
        type=int,
        default=1000,
        help="Total number of options per augmented question (must be divisible by 10).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds. Each produces its own seed_<N>/ subfolder.",
    )
    args = parser.parse_args()
    main(args)
