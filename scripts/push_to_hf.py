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
from huggingface_hub import HfApi

load_dotenv()

HF_REPO  = "disi-unibo-nlp/MMLU-1K"
BASE_DIR = "data/bench"
SEED_DIR = "seed_42"

N_OPTIONS = [100, 500, 1000]
MODES     = ["block", "sparse"]
WINDOWS   = ["0.00", "0.25", "0.50", "0.75", "1.00"]
SPLIT_MAP = {"0.00": "w0", "0.25": "w25", "0.50": "w50", "0.75": "w75", "1.00": "w100"}

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

DATASET_CARD = """\
---
language:
- en
license: apache-2.0
task_categories:
- question-answering
tags:
- multiple-choice
- reasoning
- positional-bias
- benchmark
- mmlu
- evaluation
pretty_name: MMLU-1K
size_categories:
- 10K<n<100K
configs:
- config_name: n100_block
  description: "100 options, block layout (original 10 options contiguous)"
- config_name: n100_sparse
  description: "100 options, sparse layout (original 10 options spread one-per-segment)"
- config_name: n500_block
  description: "500 options, block layout"
- config_name: n500_sparse
  description: "500 options, sparse layout"
- config_name: n1000_block
  description: "1000 options, block layout"
- config_name: n1000_sparse
  description: "1000 options, sparse layout"
---

# MMLU-1K

**MMLU-1K** is a diagnostic benchmark for evaluating LLMs under extreme multiple-choice scaling, introduced in the paper *Lost in Choice: Evaluating LLMs Under Extreme Multiple-Choice Scaling*.

It exposes a failure mode we call the **Lost-in-Choice phenomenon**: models that perform near ceiling on 10 options can collapse to near-random accuracy at 1,000 — driven by layout and position rather than knowledge.

## Dataset Description

MMLU-1K is built from **1,000 questions** drawn from MMLU-Pro across four domains (Computer Science, Health, Law, Mathematics — 250 each). Each question is expanded from its original 10 options to N ∈ {100, 500, 1000} options by injecting in-domain distractors, yielding **30,000 unique test instances**.

### Question Typology

Questions are labelled along a key axis:

- **Option-Independent** (540 questions): the correct answer can be derived from internal knowledge without inspecting the options (e.g., *"What is the capital of Italy?"*).
- **Option-Dependent** (460 questions): the correct answer only emerges through comparative analysis of the available alternatives (e.g., *"Which of the following algorithms is most efficient?"*).

| Domain           | Opt-Independent | Opt-Dependent | Total |
|------------------|:-:|:-:|:-:|
| Computer Science | 204 | 46  | 250 |
| Health           | 107 | 143 | 250 |
| Law              |  57 | 193 | 250 |
| Mathematics      | 172 |  78 | 250 |

### Spatial Layout

The original 10 options are embedded in the expanded candidate pool in two ways:

- **Block**: the 10 original options stay contiguous, shifted as a block to the target position.
- **Sparse**: the 10 original options are spread uniformly one-per-segment across the full list.

### Target Position (Window)

The gold answer is placed at five positions across the context window:

| Split | Window | Description |
|:-----:|:------:|-------------|
| `w0`  | 0%     | Gold answer at the very beginning |
| `w25` | 25%    | Gold answer at the first quarter |
| `w50` | 50%    | Gold answer at the midpoint |
| `w75` | 75%    | Gold answer at the third quarter |
| `w100`| 100%   | Gold answer at the very end |

### Distractor Quality Control

All 990,000 question-distractor pairs were screened by a dual-judge LLM pipeline (GPT-OSS-120B for CS/Law/Health; Gemini-3-Flash for Math) to eliminate coincidentally correct distractors. 51.5% of questions had at least one flagged distractor; 2,649 replacements were applied.

## Dataset Structure

```
6 configurations × 5 splits = 30 subsets (1,000 rows each)

Config       Splits
──────────   ────────────────────────────
n100_block   w0, w25, w50, w75, w100
n100_sparse  w0, w25, w50, w75, w100
n500_block   w0, w25, w50, w75, w100
n500_sparse  w0, w25, w50, w75, w100
n1000_block  w0, w25, w50, w75, w100
n1000_sparse w0, w25, w50, w75, w100
```

## Usage

```python
from datasets import load_dataset

# Load 1000-option sparse, gold at midpoint
ds = load_dataset("disi-unibo-nlp/MMLU-1K", "n1000_sparse", split="w50")

# Load all window positions for a given configuration
ds = load_dataset("disi-unibo-nlp/MMLU-1K", "n1000_sparse")
# ds["w0"], ds["w25"], ..., ds["w100"]
```

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | string | Unique question identifier |
| `question` | string | Question text |
| `options` | list[string] | Original 10 answer options |
| `answer` | string | Correct answer letter (A–J) |
| `answer_index` | int | 0-based index of the correct answer in the original 10 options |
| `category` | string | Domain: `computer science`, `health`, `law`, `math` |
| `src` | string | Source subset within MMLU-Pro |
| `option_dependent_vs_independent` | string | `option_dependent` or `option_independent` |
| `new_options` | list[string] | Full expanded option list (N options) |
| `new_answer_index` | int | 0-based index of the correct answer in `new_options` |
| `block_start_index` | int | Absolute index where the original option block begins in `new_options` |
| `window` | float | Target position as a fraction of the context window (0.0–1.0) |
| `placement_mode` | string | `block` or `sparse` |
| `n_options` | int | Total number of options (100, 500, or 1000) |

## Citation

```bibtex
@article{lost_in_choice_2025,
  title   = {Lost in Choice: Evaluating {LLMs} Under Extreme Multiple-Choice Scaling},
  year    = {2025},
}
```
"""


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
            print(f"[{config}]  {len(splits)} splits × {n_rows} rows  →  {list(splits)}")

            if not dry_run:
                dataset_dict.push_to_hub(HF_REPO, config_name=config, token=token)
                print(f"  ✓ pushed")

    if not dry_run:
        print("\nUploading dataset card (README.md)...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=DATASET_CARD.encode(),
            path_in_repo="README.md",
            repo_id=HF_REPO,
            repo_type="dataset",
            token=token,
            commit_message="Upload dataset card",
        )
        print(f"  ✓ dataset card pushed")
        print(f"\nDone. https://huggingface.co/datasets/{HF_REPO}")
    else:
        print("\nDry run complete — nothing uploaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
