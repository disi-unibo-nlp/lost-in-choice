"""
Push MMLU-1K to HuggingFace Hub as disi-unibo-nlp/MMLU-1K.

Dataset layout
──────────────
  6 configurations × 5 splits = 30 subsets

  Configurations  : n{N}_{mode}
                    n100_block  · n100_sparse
                    n500_block  · n500_sparse
                    n1000_block · n1000_sparse

  Splits          : w{pct}  (gold answer position as % of context window)
                    w0 · w25 · w50 · w75 · w100

Usage
─────
  # Requires HF_TOKEN in .env or prior `huggingface-cli login`
  python scripts/push_to_hf.py

  # Dry-run: print configs/splits without uploading
  python scripts/push_to_hf.py --dry-run
"""

import argparse
import os

from datasets import DatasetDict, load_from_disk
from dotenv import load_dotenv

load_dotenv()

HF_REPO   = "disi-unibo-nlp/MMLU-1K"
BASE_DIR  = "data/bench"
SEED_DIR  = "seed_42"

N_OPTIONS = [100, 500, 1000]
MODES     = ["block", "sparse"]
WINDOWS   = ["0.00", "0.25", "0.50", "0.75", "1.00"]
SPLIT_MAP = {"0.00": "w0", "0.25": "w25", "0.50": "w50", "0.75": "w75", "1.00": "w100"}

# Columns exposed in the public dataset (drop internal judge predictions)
KEEP = {
    "question_id",
    "question",
    "options",               # original 10 options
    "answer",                # correct letter (A–J)
    "answer_index",          # 0-based index in original 10 options
    "category",              # computer science | health | law | math
    "src",                   # source within MMLU-Pro
    "option_dependent_vs_independent",
    "new_options",           # N augmented options (the main payload)
    "new_answer_index",      # 0-based index of gold in new_options
    "block_start_index",     # absolute index where the original block starts
    "window",                # gold position as fraction [0, 1]
    "placement_mode",        # block | sparse
    "n_options",             # 100 | 500 | 1000
}


def clean(dataset):
    drop = [c for c in dataset.column_names if c not in KEEP]
    if drop:
        dataset = dataset.remove_columns(drop)
    return dataset


def main(dry_run: bool):
    token = os.environ.get("HF_TOKEN")

    for n in N_OPTIONS:
        for mode in MODES:
            config = f"n{n}_{mode}"
            splits = {}

            for w in WINDOWS:
                path = os.path.join(
                    BASE_DIR,
                    f"mmlu-pro-augmented-clean-{n}",
                    SEED_DIR,
                    f"window_{w}",
                    mode,
                )
                ds = load_from_disk(path)
                ds = clean(ds)
                splits[SPLIT_MAP[w]] = ds

            dataset_dict = DatasetDict(splits)

            n_rows = next(iter(dataset_dict.values())).num_rows
            split_names = list(dataset_dict.keys())
            print(f"[{config}]  {len(split_names)} splits × {n_rows} rows each  →  {split_names}")

            if not dry_run:
                dataset_dict.push_to_hub(HF_REPO, config_name=config, token=token)
                print(f"  ✓ pushed")

    if dry_run:
        print("\nDry run complete — nothing uploaded.")
    else:
        print(f"\nDone. Dataset available at https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print layout without uploading")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
