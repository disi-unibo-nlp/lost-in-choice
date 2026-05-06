"""
Apply distractor replacements to all (n, window, mode) variants for one seed.

The flagged_distractors.jsonl produced by retrieve_distractor_check.py is the
result of running check_distractors.py on ONE file — typically the n=1000,
window=0.00, block variant for a given seed.  That single run is sufficient
because:

  • Distractors are window-invariant (same RNG state at sample time for all
    windows within a seed).
  • Block and sparse share the same distractor set (v4 guarantee).
  • n=500 and n=100 use strict prefixes of n=1000's distractor list.

So the replacement map (old_text → new_text) derived from the 1000-option check
applies to every (n, window, mode) variant of that seed.  Distractors that
belong to the 1000-set but are absent from a smaller-N variant are silently
skipped (they simply don't appear there).

Replacements are matched by distractor TEXT, not by position index, so they
work regardless of where the distractor ends up after shuffling.

Both the JSONL flat file and the HuggingFace dataset directory are updated so
downstream scripts (run_inference.py uses load_from_disk) stay in sync.

Usage
─────
  # Update in-place (overwrites the augmented data for seed 42):
  python preprocess/apply_distractor_fixes.py \\
      --flagged-jsonl out/distractor_check/.../flagged_distractors.jsonl \\
      --seed 42

  # Write to a separate clean directory instead:
  python preprocess/apply_distractor_fixes.py \\
      --flagged-jsonl out/distractor_check/.../flagged_distractors.jsonl \\
      --seed 42 \\
      --output-base-out data/bench/mmlu-pro-augmented-clean

  # Preview without writing:
  python preprocess/apply_distractor_fixes.py \\
      --flagged-jsonl out/distractor_check/.../flagged_distractors.jsonl \\
      --seed 42 --dry-run
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from datasets import Dataset

WINDOW_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def window_label(w: float) -> str:
    return f"window_{w:.2f}"


# ---------------------------------------------------------------------------
# Load replacement map
# ---------------------------------------------------------------------------

def load_flagged(flagged_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """Return {qid: {old_text: new_text}} for questions with ≥1 flag."""
    replacements: Dict[str, Dict[str, str]] = {}
    with open(flagged_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec["question_id"])
            rmap: Dict[str, str] = {}
            for item in rec.get("flagged", []):
                rmap[item["original_text"]] = item["replacement"]
            if rmap:
                replacements[qid] = rmap
    return replacements


# ---------------------------------------------------------------------------
# Per-record replacement
# ---------------------------------------------------------------------------

def _safe_fallback(
    orig_set: Set[str],
    flagged_texts: Set[str],
    new_opts: List[str],
    used_replacements: Set[str],
    rng: random.Random,
) -> Optional[str]:
    """Return a random distractor that is safe to use as a fallback replacement.

    Safe means: not an original option, not itself flagged, not already
    used as a replacement in this question (to avoid creating new duplicates).
    Returns None if no safe candidate exists.
    """
    candidates = [
        o for o in new_opts
        if o not in orig_set
        and o not in flagged_texts
        and o not in used_replacements
    ]
    return rng.choice(candidates) if candidates else None


def apply_to_record(
    rec: dict,
    replacements: Dict[str, Dict[str, str]],
    rng: random.Random,
) -> Tuple[dict, int, int, List[str]]:
    """Apply replacement map to one record.

    When the judge's suggested replacement would collide with one of the
    original 10 options, a random safe distractor is used instead (one that
    is not flagged and not already an original option).

    Returns (updated_record, n_replaced, n_fallback, warnings).
    Original options and the gold answer are never touched.
    """
    qid = str(rec["question_id"])
    rmap = replacements.get(qid)
    if not rmap:
        return rec, 0, 0, []

    orig_set: Set[str] = set(rec["options"])
    new_opts: List[str] = list(rec["new_options"])
    flagged_texts: Set[str] = set(rmap.keys())

    n_replaced = 0
    n_fallback = 0
    warnings: List[str] = []
    used_replacements: Set[str] = set()

    for i, opt in enumerate(new_opts):
        if opt in orig_set:
            continue  # never touch original options
        if opt not in rmap:
            continue

        replacement = rmap[opt]

        if replacement in orig_set:
            # Suggested replacement collides with an original option — pick a
            # safe distractor from the pool instead.
            fallback = _safe_fallback(orig_set, flagged_texts, new_opts, used_replacements, rng)
            if fallback is None:
                warnings.append(
                    f"  [WARN] qid={qid}: no safe fallback for {opt!r} "
                    f"(judge replacement {replacement!r} collides with original) — skipped"
                )
                continue
            warnings.append(
                f"  [INFO] qid={qid}: {opt!r} → fallback {fallback!r} "
                f"(judge suggested {replacement!r} collides with original)"
            )
            replacement = fallback
            n_fallback += 1

        new_opts[i] = replacement
        used_replacements.add(replacement)
        n_replaced += 1

    rec = dict(rec)
    rec["new_options"] = new_opts
    return rec, n_replaced, n_fallback, warnings


# ---------------------------------------------------------------------------
# Per-variant processing
# ---------------------------------------------------------------------------

def process_variant(
    src_jsonl: Path,
    dst_jsonl: Path,
    dst_hf_dir: Path,
    replacements: Dict[str, Dict[str, str]],
    rng: random.Random,
    dry_run: bool,
) -> Tuple[int, int, int, int, List[str]]:
    """Process one (n, window, mode) file.

    Returns (n_questions, n_questions_modified, n_replacements_total, n_fallbacks_total, warnings).
    """
    records = []
    with open(src_jsonl) as f:
        for line in f:
            records.append(json.loads(line))

    updated_records = []
    n_modified = 0
    n_replaced_total = 0
    n_fallback_total = 0
    all_warnings: List[str] = []

    for rec in records:
        updated, n_rep, n_fb, warns = apply_to_record(rec, replacements, rng)
        updated_records.append(updated)
        all_warnings.extend(warns)
        if n_rep > 0:
            n_modified += 1
            n_replaced_total += n_rep
            n_fallback_total += n_fb

    if not dry_run:
        os.makedirs(dst_jsonl.parent, exist_ok=True)
        with open(dst_jsonl, "w") as f:
            for rec in updated_records:
                f.write(json.dumps(rec) + "\n")

        os.makedirs(dst_hf_dir, exist_ok=True)
        Dataset.from_list(updated_records).save_to_disk(str(dst_hf_dir))

    return len(records), n_modified, n_replaced_total, n_fallback_total, all_warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    flagged_jsonl = Path(args.flagged_jsonl)
    if not flagged_jsonl.exists():
        raise FileNotFoundError(f"flagged_jsonl not found: {flagged_jsonl}")

    replacements = load_flagged(flagged_jsonl)
    n_qs_with_flags = len(replacements)
    n_total_flags = sum(len(v) for v in replacements.values())
    print(f"Loaded: {flagged_jsonl}")
    print(f"  {n_qs_with_flags} questions with flagged distractors  "
          f"({n_total_flags} unique old→new mappings)\n")

    n_options_desc = sorted(args.n_options, reverse=True)
    out_base = args.output_base_out or args.output_base
    inplace = (out_base == args.output_base)

    rng = random.Random(args.seed)

    print(f"Seed        : {args.seed}")
    print(f"Sizes       : {n_options_desc}")
    print(f"Source base : {args.output_base}-<N>")
    print(f"Dest base   : {out_base}-<N>  {'(in-place)' if inplace else ''}")
    if args.dry_run:
        print("DRY RUN — no files will be written\n")
    print()

    grand_questions = 0
    grand_modified = 0
    grand_replaced = 0
    grand_fallback = 0
    all_warnings: List[str] = []

    for n in n_options_desc:
        src_root = Path(f"{args.output_base}-{n}") / f"seed_{args.seed}"
        dst_root = Path(f"{out_base}-{n}") / f"seed_{args.seed}"
        print(f"  [n={n}]")

        for window in args.windows:
            label = window_label(window)
            for mode in ("block", "sparse"):
                src_jsonl = src_root / label / f"{mode}.jsonl"
                dst_jsonl  = dst_root / label / f"{mode}.jsonl"
                dst_hf_dir = dst_root / label / mode

                if not src_jsonl.exists():
                    print(f"    [SKIP] {src_jsonl} not found")
                    continue

                n_q, n_mod, n_rep, n_fb, warns = process_variant(
                    src_jsonl, dst_jsonl, dst_hf_dir, replacements, rng, args.dry_run
                )
                all_warnings.extend(warns)
                grand_questions += n_q
                grand_modified  += n_mod
                grand_replaced  += n_rep
                grand_fallback  += n_fb

                tag = "(dry)" if args.dry_run else "saved"
                fb_tag = f", {n_fb} fallbacks" if n_fb else ""
                print(
                    f"    {label}/{mode}: {n_q} questions — "
                    f"{n_mod} modified, {n_rep} replacements{fb_tag}  [{tag}]"
                )

        print()

    info_warns = [w for w in all_warnings if "[INFO]" in w]
    real_warns = [w for w in all_warnings if "[WARN]" in w]
    if real_warns:
        print(f"{'='*60}")
        print(f"  {len(real_warns)} warnings:")
        for w in real_warns[:30]:
            print(w)
        if len(real_warns) > 30:
            print(f"  ... and {len(real_warns) - 30} more")
        print()

    print(f"{'='*60}")
    print(f"  Total questions processed : {grand_questions}")
    print(f"  Questions modified        : {grand_modified}")
    print(f"  Individual replacements   : {grand_replaced}")
    print(f"  Fallback replacements     : {grand_fallback}")
    if not args.dry_run:
        print(f"  Output written to        : {out_base}-<N>/seed_{args.seed}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply distractor replacements to all (n, window, mode) variants for one seed."
    )
    parser.add_argument(
        "--flagged-jsonl", required=True,
        help="flagged_distractors.jsonl produced by retrieve_distractor_check.py",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="Seed whose variants to update (run once per seed)",
    )
    parser.add_argument(
        "--output-base", type=str, default="data/bench/mmlu-pro-augmented",
        help="Source base prefix (matches --output-base from augment_final_dataset_v4.py)",
    )
    parser.add_argument(
        "--output-base-out", type=str, default=None,
        help="Destination base prefix. Omit to update in-place.",
    )
    parser.add_argument(
        "--n-options", type=int, nargs="+", default=[100, 500, 1000],
    )
    parser.add_argument(
        "--windows", type=float, nargs="+", default=WINDOW_POSITIONS,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be changed without writing any files",
    )
    args = parser.parse_args()
    main(args)
