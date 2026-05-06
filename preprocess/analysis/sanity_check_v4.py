"""
Sanity-check for augment_final_dataset_v4.py output.

Checks per question per (seed, window)
───────────────────────────────────────
  (A) len(new_options) == n
  (B) new_options[new_answer_index] == options[answer_index]
  (C) new_answer_index == block_start_index + answer_index  (both modes)
  (D) block mode  : new_options[block_start : block_start+10] == options
  (E) sparse mode : each original option appears exactly once in new_options
  (F) sparse mode : no two original options share the same segment
  (G) gold index identical between block and sparse for the same n
  (H) block and sparse have the same set of distractor texts for the same n
  (I) NEW (v4): distractors(n_small) ⊆ distractors(n_large) for all size pairs
  (J) NEW (v4): |union of distractors across all n| ≤ max_n − 10
  (K) NEW (v4): distractor set is identical across ALL window positions
               for a given (seed, n, question).  The RNG is re-seeded with the
               same value for every window and window never changes the number
               of RNG calls, so this must always hold.

Usage:
  python3 preprocess/analysis/sanity_check_v4.py \\
      --output-base data/bench/mmlu-pro-augmented \\
      --n-options 100 500 1000 \\
      --seeds 42 123 456 \\
      --print-samples 2
"""

import argparse
import os
import sys
from typing import Dict, List, Set, Tuple

from datasets import load_from_disk

N_ORIG = 10
WINDOW_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def window_label(w: float) -> str:
    return f"window_{w:.2f}"


def load_variant(output_base: str, seed: int, n: int, window: float, mode: str):
    path = os.path.join(f"{output_base}-{n}", f"seed_{seed}", window_label(window), mode)
    if not os.path.isdir(path):
        return None
    return load_from_disk(path)


def distractor_set(new_options: list, orig_options: list) -> Set[str]:
    orig_set = set(orig_options)
    return {opt for opt in new_options if opt not in orig_set}


# ---------------------------------------------------------------------------
# Per-question checks (A–H): identical constraints to v3 + size-aware
# ---------------------------------------------------------------------------

def check_single_variant(
    ex_block,
    ex_sparse,
    n: int,
    errors: list,
    qid: str,
) -> Tuple[Set[str], Set[str]]:
    """Run checks A–H for one (n, window) pair.

    Returns (dist_block, dist_sparse) so the caller can use them for
    cross-n checks without recomputing.
    """
    seg_size = n // N_ORIG
    orig_opts = list(ex_block["options"])
    ans_idx = ex_block["answer_index"]
    correct_text = orig_opts[ans_idx]

    for mode, ex in [("block", ex_block), ("sparse", ex_sparse)]:
        new_opts = ex["new_options"]
        new_idx = ex["new_answer_index"]
        b_start = ex["block_start_index"]

        # (A) length
        if len(new_opts) != n:
            errors.append(f"[A/{mode}] qid={qid}: len={len(new_opts)} != {n}")

        # (B) correct text at declared index
        if new_opts[new_idx] != correct_text:
            errors.append(
                f"[B/{mode}] qid={qid}: new_opts[{new_idx}]={new_opts[new_idx]!r} "
                f"!= correct={correct_text!r}"
            )

        # (C) gold == block_start + answer_index
        expected_gold = b_start + ans_idx
        if new_idx != expected_gold:
            errors.append(
                f"[C/{mode}] qid={qid}: new_answer_index={new_idx} != "
                f"block_start({b_start}) + answer_index({ans_idx}) = {expected_gold}"
            )

    # (D) block: original options contiguous in original order
    block_opts = ex_block["new_options"]
    b_start_b = ex_block["block_start_index"]
    if block_opts[b_start_b: b_start_b + N_ORIG] != orig_opts:
        errors.append(
            f"[D/block] qid={qid}: slice [{b_start_b}:{b_start_b+N_ORIG}] != original options"
        )

    # (E) sparse: each original appears exactly once
    sparse_opts = ex_sparse["new_options"]
    for orig in orig_opts:
        count = sparse_opts.count(orig)
        if count != 1:
            errors.append(
                f"[E/sparse] qid={qid}: original option {orig!r} appears {count} times"
            )

    # (F) sparse: original options in distinct segments
    orig_segs = [sparse_opts.index(o) // seg_size for o in orig_opts]
    if len(set(orig_segs)) != N_ORIG:
        errors.append(
            f"[F/sparse] qid={qid}: original options share segments — {orig_segs}"
        )

    # (G) gold index identical between block and sparse
    if ex_block["new_answer_index"] != ex_sparse["new_answer_index"]:
        errors.append(
            f"[G] qid={qid} n={n}: block gold={ex_block['new_answer_index']} "
            f"!= sparse gold={ex_sparse['new_answer_index']}"
        )

    dist_b = distractor_set(ex_block["new_options"], orig_opts)
    dist_s = distractor_set(ex_sparse["new_options"], orig_opts)

    # (H) block and sparse have same distractor set
    if dist_b != dist_s:
        only_b = dist_b - dist_s
        only_s = dist_s - dist_b
        errors.append(
            f"[H] qid={qid} n={n}: {len(only_b)} only in block, {len(only_s)} only in sparse"
        )

    return dist_b, dist_s


# ---------------------------------------------------------------------------
# Cross-n checks (I, J)
# ---------------------------------------------------------------------------

def check_cross_n(
    dist_by_n: Dict[int, Set[str]],
    n_options_desc: List[int],
    max_n: int,
    errors: list,
    qid: str,
):
    # (I) subset property
    for i in range(len(n_options_desc) - 1):
        n_large = n_options_desc[i]
        n_small = n_options_desc[i + 1]
        if not dist_by_n[n_small] <= dist_by_n[n_large]:
            extra = dist_by_n[n_small] - dist_by_n[n_large]
            errors.append(
                f"[I] qid={qid}: distractors({n_small}) ⊄ distractors({n_large}) "
                f"— {len(extra)} extra items, e.g. {sorted(extra)[:2]}"
            )

    # (J) total distinct ≤ max_n − 10
    union_dist = set().union(*dist_by_n.values())
    max_allowed = max_n - N_ORIG
    if len(union_dist) > max_allowed:
        errors.append(
            f"[J] qid={qid}: union distractors={len(union_dist)} > {max_allowed}"
        )


# ---------------------------------------------------------------------------
# Sample printer
# ---------------------------------------------------------------------------

def print_sample(
    data_by_n: Dict[int, Tuple],  # n -> (ex_block, ex_sparse)
    n_options_desc: List[int],
    sample_idx: int,
):
    n_ref = n_options_desc[0]
    ex_b, ex_s = data_by_n[n_ref]
    orig_opts = list(ex_b["options"])
    ans_idx = ex_b["answer_index"]
    correct_text = orig_opts[ans_idx]

    q_text = ex_b.get("question", "")
    q_preview = q_text[:97] + ("..." if len(q_text) > 97 else "")

    print(f"\n  --- Sample {sample_idx} ---")
    print(f"  question_id : {ex_b.get('question_id', 'N/A')}")
    print(f"  category    : {ex_b.get('category', 'N/A')}")
    print(f"  question    : {q_preview}")
    print(f"  correct     : {correct_text!r}  (original index {ans_idx})")

    for n in n_options_desc:
        ex_block, ex_sparse = data_by_n[n]
        seg_size = n // N_ORIG
        b_start = ex_block["block_start_index"]
        gold_b = ex_block["new_answer_index"]
        gold_s = ex_sparse["new_answer_index"]
        dist_b = distractor_set(ex_block["new_options"], orig_opts)
        dist_s = distractor_set(ex_sparse["new_options"], orig_opts)
        gold_match = "✓" if gold_b == gold_s else "✗ MISMATCH"
        dist_match = "✓ identical (%d)" % len(dist_b) if dist_b == dist_s else "✗ DIFFER"
        print(
            f"\n  [n={n}]  block_start={b_start}  gold_block={gold_b}  "
            f"gold_sparse={gold_s} {gold_match}  distractors {dist_match}"
        )
        # Show block region (first 4 options)
        block_opts = ex_block["new_options"]
        print(f"    block region new_opts[{b_start}:{b_start+N_ORIG}] (first 4):")
        for i, opt in enumerate(block_opts[b_start: b_start + min(4, N_ORIG)]):
            marker = " ← GOLD" if (b_start + i) == gold_b else ""
            print(f"      [{b_start+i:5d}] {opt[:65]!r}{marker}")

        # Show sparse original placement (first 4)
        sparse_opts = ex_sparse["new_options"]
        print(f"    sparse originals (seg_size={seg_size}, first 4):")
        for orig in orig_opts[:4]:
            idx = sparse_opts.index(orig)
            seg = idx // seg_size
            marker = " ← GOLD" if idx == gold_s else ""
            print(f"      seg {seg:2d}  [{idx:5d}] {orig[:60]!r}{marker}")

    # Subset summary across n values
    if len(n_options_desc) > 1:
        print(f"\n  Subset check across sizes:")
        dists = {n: distractor_set(data_by_n[n][0]["new_options"], orig_opts) for n in n_options_desc}
        for i in range(len(n_options_desc) - 1):
            nl, ns = n_options_desc[i], n_options_desc[i + 1]
            ok = dists[ns] <= dists[nl]
            print(f"    distractors({ns}) ⊆ distractors({nl}) : {'✓' if ok else '✗ FAIL'}")


# ---------------------------------------------------------------------------
# Cross-window check (K)
# ---------------------------------------------------------------------------

def check_cross_window(
    cross_window_dists: Dict[int, Dict[int, Dict[str, Set[str]]]],
    qids: Dict[int, str],
    n_options_desc: List[int],
    expected_windows: List[str],
) -> List[str]:
    """Return error strings for any (seed, n, question) where the distractor
    set differs across window positions."""
    errors = []
    ref_label = expected_windows[0]
    for n in n_options_desc:
        for qi, win_dists in cross_window_dists[n].items():
            if ref_label not in win_dists:
                continue  # reference window was missing; already flagged
            ref_dist = win_dists[ref_label]
            for wlabel, dist in win_dists.items():
                if wlabel == ref_label:
                    continue
                if dist != ref_dist:
                    only_ref = ref_dist - dist
                    only_w   = dist - ref_dist
                    errors.append(
                        f"[K] n={n} qid={qids.get(qi, qi)}: distractors differ "
                        f"between {ref_label} and {wlabel} — "
                        f"{len(only_ref)} only in ref, {len(only_w)} only in {wlabel}"
                    )
    return errors


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(args):
    n_options_desc = sorted(args.n_options, reverse=True)
    max_n = n_options_desc[0]
    windows = args.windows
    win_labels = [window_label(w) for w in windows]

    total_questions = 0
    total_errors = 0

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  Seed {seed}")
        print(f"{'='*60}")

        # cross_window_dists[n][qi][win_label] = distractor_set
        # Populated during the window loop; checked after all windows are done.
        cross_window_dists: Dict[int, Dict[int, Dict[str, Set[str]]]] = {
            n: {} for n in n_options_desc
        }
        qids: Dict[int, str] = {}  # qi → question_id string
        n_q_global: int = 0
        samples_printed_global = 0

        for window in windows:
            label = window_label(window)

            # Load all (n, mode) datasets for this (seed, window)
            datasets: Dict[int, Dict[str, object]] = {}
            missing = False
            for n in n_options_desc:
                datasets[n] = {}
                for mode in ("block", "sparse"):
                    ds = load_variant(args.output_base, seed, n, window, mode)
                    if ds is None:
                        print(
                            f"  [MISSING] seed={seed} n={n} {label}/{mode} — "
                            f"path not found, skipping window"
                        )
                        missing = True
                    else:
                        datasets[n][mode] = ds

            if missing:
                total_errors += 1
                continue

            # All sizes must have the same number of questions
            n_q = len(datasets[n_options_desc[0]]["block"])
            n_q_global = n_q
            for n in n_options_desc:
                for mode in ("block", "sparse"):
                    if len(datasets[n][mode]) != n_q:
                        print(
                            f"  [ERROR] n={n} {mode} has {len(datasets[n][mode])} rows, "
                            f"expected {n_q}"
                        )
                        total_errors += 1

            errors: List[str] = []

            for qi in range(n_q):
                qid = str(datasets[n_options_desc[0]]["block"][qi].get("question_id", qi))
                qids[qi] = qid

                dist_by_n: Dict[int, Set[str]] = {}
                data_for_sample: Dict[int, Tuple] = {}

                for n in n_options_desc:
                    ex_b = datasets[n]["block"][qi]
                    ex_s = datasets[n]["sparse"][qi]
                    data_for_sample[n] = (ex_b, ex_s)

                    dist_b, _ = check_single_variant(ex_b, ex_s, n, errors, qid)
                    dist_by_n[n] = dist_b

                    # Collect for cross-window check (K)
                    cross_window_dists[n].setdefault(qi, {})[label] = dist_b

                check_cross_n(dist_by_n, n_options_desc, max_n, errors, qid)

                if args.print_samples and samples_printed_global < args.print_samples:
                    if samples_printed_global == 0:
                        print(f"\n  [{label}] Sample questions:")
                    print_sample(data_for_sample, n_options_desc, samples_printed_global + 1)
                    samples_printed_global += 1

            total_questions += n_q

            if errors:
                total_errors += len(errors)
                print(f"\n  [{label}]  {n_q} questions — {len(errors)} ERRORS:")
                for e in errors[:20]:
                    print(f"    ✗ {e}")
                if len(errors) > 20:
                    print(f"    ... and {len(errors) - 20} more")
            else:
                parts = []
                for n in n_options_desc:
                    b_start_val = datasets[n]["block"][0]["block_start_index"]
                    gold_idxs = [r["new_answer_index"] for r in datasets[n]["block"]]
                    parts.append(
                        f"n={n} bs={b_start_val} gold[{min(gold_idxs)},{max(gold_idxs)}]"
                    )
                print(
                    f"  [{label}]  {n_q} questions  ✓ all checks A–J passed  "
                    + "  |  ".join(parts)
                )

        # --- Check K: distractor invariance across window positions ---
        print(f"\n  [K] Cross-window distractor invariance (seed={seed}) ...")
        k_errors = check_cross_window(cross_window_dists, qids, n_options_desc, win_labels)
        if k_errors:
            total_errors += len(k_errors)
            print(f"  [K]  {len(k_errors)} FAILURES:")
            for e in k_errors[:20]:
                print(f"    ✗ {e}")
            if len(k_errors) > 20:
                print(f"    ... and {len(k_errors) - 20} more")
        else:
            print(
                f"  [K]  ✓ distractors identical across all {len(windows)} windows "
                f"for all {n_q_global} questions × {len(n_options_desc)} sizes"
            )

    print(f"\n{'='*60}")
    if total_errors == 0:
        print(f"  ALL CHECKS PASSED  ({total_questions} questions across all variants)")
    else:
        print(f"  FAILED: {total_errors} errors across {total_questions} questions")
    print(f"{'='*60}\n")

    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check augment_final_dataset_v4.py output."
    )
    parser.add_argument("--output-base", type=str,
                        default="data/bench/mmlu-pro-augmented",
                        help="Base prefix matching --output-base from augment_final_dataset_v4.py. "
                             "Each N is loaded from <output-base>-N/.")
    parser.add_argument("--n-options", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--windows", type=float, nargs="+", default=WINDOW_POSITIONS)
    parser.add_argument("--print-samples", type=int, default=2, metavar="N",
                        help="Print N sample questions per window (0 to disable)")
    args = parser.parse_args()
    sys.exit(main(args))
