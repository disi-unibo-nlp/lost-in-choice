"""
Augment the final 1000-question dataset to multiple N-option sizes.  (v4)

Changes from v3
───────────────
• --n-options now accepts a list of sizes (e.g. --n-options 100 500 1000).
• Distractors are sampled ONCE per question per (seed, window) for the LARGEST
  requested N.  Smaller-N variants reuse a strict PREFIX of that sample:

      distractors(N=100) ⊆ distractors(N=500) ⊆ distractors(N=1000)

  At most max_N − 10 distinct distractors ever appear for a given question
  across all size configurations.
• Block and sparse still share the same distractor list for a given N
  (inherited from v3).
• The shared RNG is consumed in a fixed order (sizes descending, then
  block before sparse) so results are fully reproducible.

Output layout
─────────────
Each N gets its own root directory, matching the v3 convention:

  data/bench/mmlu-pro-augmented-<N>/seed_<seed>/window_<pos>/block/
  data/bench/mmlu-pro-augmented-<N>/seed_<seed>/window_<pos>/block.jsonl
  data/bench/mmlu-pro-augmented-<N>/seed_<seed>/window_<pos>/sparse/
  data/bench/mmlu-pro-augmented-<N>/seed_<seed>/window_<pos>/sparse.jsonl

The base prefix (default "data/bench/mmlu-pro-augmented") is set via
--output-base; the per-N root is "{output_base}-{N}".

Each record carries a new field  n_options: int  (the size for that variant).
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
N_ORIG = 10


# ---------------------------------------------------------------------------
# Core helpers  (unchanged from v3 except n is now explicit everywhere)
# ---------------------------------------------------------------------------

def block_start(window: float, n: int, b: int = N_ORIG) -> int:
    return int(round(window * (n - b)))


def sample_distractors(
    pool: List[str],
    exclude: List[str],
    needed: int,
    rng: random.Random,
) -> List[str]:
    exclude_set = set(exclude)
    candidates = [t for t in pool if t not in exclude_set]
    if len(candidates) >= needed:
        return rng.sample(candidates, needed)
    result = list(candidates)
    result += rng.choices(candidates if candidates else pool, k=needed - len(result))
    return result


def augment_block(
    original_options: List[str],
    correct_text: str,
    distractors: List[str],
    window: float,
    rng: random.Random,
    n: int,
) -> Tuple[List[str], int, int]:
    b_start = block_start(window, n)
    others = list(distractors)
    rng.shuffle(others)
    block = list(original_options)
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
    n: int,
) -> Tuple[List[str], int, int]:
    assert n % N_ORIG == 0, f"n={n} must be divisible by N_ORIG={N_ORIG}"
    b_start = block_start(window, n)
    gold_in_block = list(original_options).index(correct_text)
    gold_idx = b_start + gold_in_block

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
# Dataset-level helpers
# ---------------------------------------------------------------------------

def build_category_pools(source_dataset) -> Dict[str, List[str]]:
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


def augment_dataset_all_sizes(
    dataset,
    category_pools: Dict[str, List[str]],
    window: float,
    seed: int,
    n_options_desc: List[int],
) -> Dict[int, Tuple[Dataset, Dataset]]:
    """Single pass over the dataset for one (seed, window).

    Samples max_N − 10 distractors once per question.  Each smaller-N variant
    takes a prefix of that list, guaranteeing the subset property.
    Block and sparse use the same prefix for a given N.

    Returns {n: (block_dataset, sparse_dataset)}.
    """
    max_n = n_options_desc[0]
    n_distractors_max = max_n - N_ORIG

    rng = random.Random(seed)
    records: Dict[int, Dict[str, list]] = {
        n: {"block": [], "sparse": []} for n in n_options_desc
    }

    for ex in dataset:
        ans_idx = ex["answer_index"]
        correct_text = ex["options"][ans_idx]
        orig_options = list(ex["options"])
        cat = ex["category"]
        pool = category_pools.get(cat, [])

        # Sample the full distractor budget for the largest N — one call, one RNG draw
        all_distractors = sample_distractors(
            pool, exclude=orig_options, needed=n_distractors_max, rng=rng
        )

        # Collect distractor sets per n for the post-hoc subset assertion
        dist_sets: Dict[int, set] = {}
        orig_set = set(orig_options)

        # Process sizes descending so each smaller N uses an already-seen prefix
        for n in n_options_desc:
            n_needed = n - N_ORIG
            distractors = all_distractors[:n_needed]  # strict prefix → subset guaranteed

            new_opts_b, new_ans_b, b_start_b = augment_block(
                orig_options, correct_text, distractors, window, rng, n
            )
            new_opts_s, new_ans_s, b_start_s = augment_sparse(
                orig_options, correct_text, distractors, window, rng, n
            )

            dist_in_block  = set(new_opts_b) - orig_set
            dist_in_sparse = set(new_opts_s) - orig_set
            assert dist_in_block == dist_in_sparse, (
                f"qid={ex.get('question_id','?')} n={n}: "
                "distractor sets differ between block and sparse"
            )
            assert new_opts_b[new_ans_b] == correct_text, (
                f"qid={ex.get('question_id','?')} n={n}: wrong gold in block"
            )
            assert new_opts_s[new_ans_s] == correct_text, (
                f"qid={ex.get('question_id','?')} n={n}: wrong gold in sparse"
            )
            assert new_ans_b == new_ans_s, (
                f"qid={ex.get('question_id','?')} n={n}: "
                f"gold index differs block={new_ans_b} sparse={new_ans_s}"
            )
            dist_sets[n] = dist_in_block

            base = dict(ex)
            records[n]["block"].append({
                **base,
                "new_options":       new_opts_b,
                "new_answer_index":  new_ans_b,
                "block_start_index": b_start_b,
                "window":            window,
                "placement_mode":    "block",
                "n_options":         n,
            })
            records[n]["sparse"].append({
                **base,
                "new_options":       new_opts_s,
                "new_answer_index":  new_ans_s,
                "block_start_index": b_start_s,
                "window":            window,
                "placement_mode":    "sparse",
                "n_options":         n,
            })

        # Verify subset property: distractors(n_small) ⊆ distractors(n_large)
        for i in range(len(n_options_desc) - 1):
            n_large = n_options_desc[i]
            n_small = n_options_desc[i + 1]
            assert dist_sets[n_small] <= dist_sets[n_large], (
                f"qid={ex.get('question_id','?')}: "
                f"distractors({n_small}) ⊄ distractors({n_large})"
            )

    return {
        n: (
            Dataset.from_list(records[n]["block"]),
            Dataset.from_list(records[n]["sparse"]),
        )
        for n in n_options_desc
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    n_options_desc = sorted(args.n_options, reverse=True)
    max_n = n_options_desc[0]

    for n in n_options_desc:
        if n % N_ORIG != 0:
            raise ValueError(
                f"All --n-options values must be divisible by {N_ORIG}; got {n}"
            )

    print(f"Loading final dataset from: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    print(f"  {len(ds)} questions loaded")

    print(f"\nLoading source dataset from: {args.source_dataset_path}")
    source_ds = load_from_disk(args.source_dataset_path)
    print(f"  {len(source_ds)} questions in source")

    category_pools = build_category_pools(source_ds)

    n_variants = len(args.seeds) * len(WINDOW_POSITIONS) * 2 * len(n_options_desc)
    print(
        f"Generating: {len(args.seeds)} seeds × {len(WINDOW_POSITIONS)} windows × "
        f"2 modes × {len(n_options_desc)} sizes = {n_variants} total variants\n"
        f"  n_options (desc): {n_options_desc}\n"
        f"  max distractors per question: {max_n - N_ORIG}\n"
    )

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        for window in WINDOW_POSITIONS:
            label = f"window_{window:.2f}"
            print(f"\n  [{label}]")

            results = augment_dataset_all_sizes(
                ds,
                category_pools=category_pools,
                window=window,
                seed=seed,
                n_options_desc=n_options_desc,
            )

            for n, (block_ds, sparse_ds) in results.items():
                out_root = f"{args.output_base}-{n}"
                seed_dir = os.path.join(out_root, f"seed_{seed}")
                b_start_val = block_start(window, n)
                print(
                    f"    [n={n:>5}]  block_start={b_start_val:>5}  "
                    f"(gold range [{b_start_val}, {b_start_val + N_ORIG - 1}])"
                )
                for mode, aug_ds in [("block", block_ds), ("sparse", sparse_ds)]:
                    indices = [r["new_answer_index"] for r in aug_ds]
                    print(
                        f"      [{mode}]  gold — min={min(indices)}  "
                        f"max={max(indices)}  mean={sum(indices)/len(indices):.1f}"
                    )
                    out_dir = os.path.join(seed_dir, label, mode)
                    os.makedirs(out_dir, exist_ok=True)
                    aug_ds.save_to_disk(out_dir)

                    jsonl_path = os.path.join(seed_dir, label, f"{mode}.jsonl")
                    aug_ds.to_json(jsonl_path, orient="records", lines=True)
                    print(f"      Saved → {out_dir}/")

    roots = [f"{args.output_base}-{n}" for n in n_options_desc]
    print(f"\nAll {n_variants} variants saved under: {', '.join(roots)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment the final dataset to multiple N-option sizes (v4)."
    )
    parser.add_argument("--dataset-path", type=str,
                        default="data/bench/mmlu-pro-augmented-10")
    parser.add_argument("--source-dataset-path", type=str,
                        default="mmlu_pro_downsampled_2858")
    parser.add_argument("--output-base", type=str,
                        default="data/bench/mmlu-pro-augmented",
                        help="Base prefix for per-N output dirs. "
                             "Each size N writes to <output-base>-N/.")
    parser.add_argument("--n-options", type=int, nargs="+", default=[100, 500, 1000],
                        help="List of option-count sizes. Must all be divisible by 10.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = parser.parse_args()
    main(args)
