"""Explore augmented MMLU-Pro datasets: print schema, stats, and a formatted first example."""

import argparse
from datasets import load_from_disk


def format_options(options, answer_index=None, max_show=20):
    """Format options list, truncating if very long."""
    n = len(options)
    lines = []
    show = min(n, max_show)
    for i, opt in enumerate(options[:show]):
        marker = " <-- GOLD" if i == answer_index else ""
        label = chr(ord("A") + i)
        lines.append(f"  {label}. {opt}{marker}")
    if n > max_show:
        lines.append(f"  ... ({n - max_show} more options not shown)")
    return "\n".join(lines)


def print_example(row, idx=0):
    print(f"\n{'='*70}")
    print(f"EXAMPLE #{idx}  (question_id={row['question_id']})")
    print(f"{'='*70}")
    print(f"Category   : {row['category']}")
    print(f"Src        : {row['src']}")
    print(f"Label type : {row.get('option_dependent_vs_independent', 'N/A')}")
    print(f"Vote method: {row.get('vote_method', 'N/A')}")
    print(f"Window     : {row.get('window', 'N/A')}")
    print(f"Placement  : {row.get('placement_mode', 'N/A')}")
    print()
    print("QUESTION:")
    print(f"  {row['question']}")
    print()

    # Original options (10)
    orig_opts = row["options"]
    orig_ans_idx = row["answer_index"]
    orig_ans_letter = row["answer"]
    print(f"ORIGINAL OPTIONS ({len(orig_opts)}) — gold: {orig_ans_letter} (index {orig_ans_idx})")
    print(format_options(orig_opts, orig_ans_idx, max_show=len(orig_opts)))
    print()

    # Augmented options (1000)
    new_opts = row["new_options"]
    new_ans_idx = row["new_answer_index"]
    new_ans_letter = chr(ord("A") + new_ans_idx) if new_ans_idx < 26 else f"[{new_ans_idx}]"
    print(f"AUGMENTED OPTIONS ({len(new_opts)}) — gold: {new_ans_letter} (index {new_ans_idx})")
    print(format_options(new_opts, new_ans_idx, max_show=20))
    print()

    print("GOLD LABEL SUMMARY:")
    print(f"  Original : answer='{orig_ans_letter}', answer_index={orig_ans_idx}")
    print(f"  Augmented: new_answer_index={new_ans_idx}  (letter '{new_ans_letter}' if alphabetic)")
    print(f"  Gold text: \"{new_opts[new_ans_idx]}\"")

    # Model predictions if present
    pred_keys = [k for k in row.keys() if k.startswith("pred_")]
    if pred_keys:
        print()
        print("MODEL PREDICTIONS:")
        for k in pred_keys:
            print(f"  {k[5:]}: {row[k]}")


def print_dataset_info(ds, path):
    print(f"\n{'='*70}")
    print(f"DATASET: {path}")
    print(f"  Rows    : {len(ds)}")
    print(f"  Columns : {ds.column_names}")
    if "category" in ds.column_names:
        from collections import Counter
        cats = Counter(ds["category"])
        print(f"  Categories ({len(cats)}):")
        for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"    {cat}: {cnt}")
    if "option_dependent_vs_independent" in ds.column_names:
        from collections import Counter
        labels = Counter(ds["option_dependent_vs_independent"])
        print(f"  option_dependent_vs_independent: {dict(labels)}")
    if "new_options" in ds.column_names:
        sample_len = len(ds[0]["new_options"])
        print(f"  Augmented options count (first row): {sample_len}")


def main():
    parser = argparse.ArgumentParser(description="Explore MMLU-Pro augmented datasets")
    parser.add_argument(
        "--path",
        default="data/bench/mmlu-pro-augmented-1000/seed_42/window_1.00/sparse",
        help="Path to HuggingFace dataset directory",
    )
    parser.add_argument("--index", type=int, default=0, help="Which example to print (default: 0)")
    parser.add_argument("--stats-only", action="store_true", help="Print stats only, no example")
    args = parser.parse_args()

    ds = load_from_disk(args.path)
    print_dataset_info(ds, args.path)

    if not args.stats_only:
        row = ds[args.index]
        print_example(row, idx=args.index)


if __name__ == "__main__":
    main()
