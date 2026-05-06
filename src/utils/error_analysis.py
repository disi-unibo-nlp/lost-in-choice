"""
Error analysis: compare model predictions across baseline (10 opts) and two augmented
conditions (e.g. 1000 opts, window=0.75, block vs. sparse) on questions the model got
right at baseline.

For each baseline-correct question the script classifies augmented-condition failures as:
  E1 — Distractor Absorption : model picked a synthetic distractor (hit_original_option=False)
  E2 — Original Mis-selection: model picked a wrong original option (hit_original_option=True)
    E2b_earlier: the wrong original sits before the gold in the list (primacy within originals)
    E2a_later : the wrong original sits after  the gold (more likely semantic confusion)

Outputs
-------
  <output-dir>/comparison.csv         — one row per baseline-correct question
  <output-dir>/qualitative_sample.txt — human-readable sample of "both wrong" cases

Usage
-----
    python src/utils/error_analysis.py \\
        --baseline-pred out/completions/gpt-oss-120b/mmlu-pro-augmented-10__baseline/2026-04-16_22-44-00 \\
        --block-pred    out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__seed_42__window_0.75__block/2026-04-23_15-23-56 \\
        --sparse-pred   out/completions/gpt-oss-120b/mmlu-pro-augmented-1000__seed_42__window_0.75__sparse/2026-04-23_15-26-13 \\
        --baseline-data data/bench/mmlu-pro-augmented-10.jsonl \\
        --block-data    data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl \\
        --sparse-data   data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/sparse.jsonl \\
        --output-dir    out/error_analysis/gpt-oss-120b__window0.75
"""

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> Dict[str, Any]:
    records = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            records[str(d["question_id"])] = d
    return records


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_error(
    pred: Dict, aug: Dict
) -> Tuple[str, Optional[str], Optional[int]]:
    """Return (error_type, predicted_text, predicted_index).

    error_type is one of: 'correct' | 'E1' | 'E2'
    """
    pred_idx = pred["predicted_index"]
    pred_text = str(aug["new_options"][pred_idx])

    if pred["correct"]:
        return "correct", pred_text, pred_idx

    error_type = "E2" if pred["hit_original_option"] else "E1"
    return error_type, pred_text, pred_idx


def e2_subtype(pred_idx: int, gold_idx: int) -> str:
    """Distinguish primacy-within-originals (E2b) from later-original confusion (E2a)."""
    return "E2b_earlier" if pred_idx < gold_idx else "E2a_later"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def wrap(text: str, width: int = 70) -> str:
    return "\n    ".join(textwrap.wrap(str(text), width))


SEP = "-" * 70


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pred", required=True,
                        help="Dir containing predictions.jsonl for the 10-option baseline")
    parser.add_argument("--block-pred", required=True,
                        help="Dir containing predictions.jsonl for the block augmented run")
    parser.add_argument("--sparse-pred", required=True,
                        help="Dir containing predictions.jsonl for the sparse augmented run")
    parser.add_argument("--baseline-data", required=True,
                        help="mmlu-pro-augmented-10.jsonl benchmark file")
    parser.add_argument("--block-data", required=True,
                        help="Augmented dataset jsonl for the block condition")
    parser.add_argument("--sparse-data", required=True,
                        help="Augmented dataset jsonl for the sparse condition")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=20,
                        help="Number of 'both wrong' questions to show qualitatively")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    print("Loading predictions …")
    baseline_preds = load_jsonl(Path(args.baseline_pred) / "predictions.jsonl")
    block_preds    = load_jsonl(Path(args.block_pred)    / "predictions.jsonl")
    sparse_preds   = load_jsonl(Path(args.sparse_pred)   / "predictions.jsonl")

    print("Loading datasets …")
    baseline_data = load_jsonl(Path(args.baseline_data))
    block_data    = load_jsonl(Path(args.block_data))
    sparse_data   = load_jsonl(Path(args.sparse_data))

    # --- intersect: baseline-correct questions present in all conditions ---
    correct_at_baseline = {qid for qid, p in baseline_preds.items() if p["correct"]}
    common = correct_at_baseline & block_preds.keys() & sparse_preds.keys()
    print(f"\nBaseline correct       : {len(correct_at_baseline)}")
    print(f"In all three conditions: {len(common)}")

    # --- build comparison table ---
    rows = []
    for qid in sorted(common):
        base  = baseline_data.get(qid, {})
        b_aug = block_data.get(qid)
        s_aug = sparse_data.get(qid)
        if not b_aug or not s_aug:
            continue

        b_err, b_pred_text, b_pred_idx = classify_error(block_preds[qid],  b_aug)
        s_err, s_pred_text, s_pred_idx = classify_error(sparse_preds[qid], s_aug)

        gold_idx  = b_aug["new_answer_index"]
        gold_text = str(b_aug["new_options"][gold_idx])

        b_sub = e2_subtype(b_pred_idx, gold_idx) if b_err == "E2" else None
        s_sub = e2_subtype(s_pred_idx, s_aug["new_answer_index"]) if s_err == "E2" else None

        rows.append({
            "question_id"     : qid,
            "category"        : base.get("category", ""),
            "opt_dep"         : base.get("option_dependent_vs_independent", ""),
            "gold_index"      : gold_idx,
            "gold_text"       : gold_text,
            "block_error"     : b_err,
            "block_pred_idx"  : b_pred_idx,
            "block_pred_text" : b_pred_text,
            "block_e2_sub"    : b_sub,
            "sparse_error"    : s_err,
            "sparse_pred_idx" : s_pred_idx,
            "sparse_pred_text": s_pred_text,
            "sparse_e2_sub"   : s_sub,
            "question"        : base.get("question", ""),
        })

    df = pd.DataFrame(rows)
    total = len(df)

    # --- save CSV (no question text to keep it small) ---
    csv_path = out_dir / "comparison.csv"
    df.drop(columns=["question"]).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({total} rows)")

    # --- aggregate stats ---
    print(f"\n{'='*60}")
    print(f"AGGREGATE  (baseline-correct, n={total})")
    print(f"{'='*60}")

    for cond, err_col, sub_col in [
        ("Block",  "block_error",  "block_e2_sub"),
        ("Sparse", "sparse_error", "sparse_e2_sub"),
    ]:
        counts = df[err_col].value_counts()
        print(f"\n{cond}:")
        for label in ["correct", "E1", "E2"]:
            n = counts.get(label, 0)
            print(f"  {label:10s}: {n:4d}  ({100*n/total:.1f}%)")
        if cond == "Sparse":
            sub = df[df[sub_col].notna()][sub_col].value_counts()
            print(f"  E2 subtype (sparse):")
            for k, v in sub.items():
                print(f"    {k}: {v}  ({100*v/total:.1f}%)")

    print(f"\n{'='*60}")
    print("BLOCK error  ×  category  ×  opt_dep")
    print(df.groupby(["category", "opt_dep"])["block_error"]
            .value_counts().unstack(fill_value=0).to_string())

    print(f"\n{'='*60}")
    print("SPARSE error  ×  category  ×  opt_dep")
    print(df.groupby(["category", "opt_dep"])["sparse_error"]
            .value_counts().unstack(fill_value=0).to_string())

    # --- both-wrong subset ---
    both_wrong = df[(df["block_error"] != "correct") & (df["sparse_error"] != "correct")]
    print(f"\n{'='*60}")
    print(f"BOTH WRONG: {len(both_wrong)} / {total}  ({100*len(both_wrong)/total:.1f}%)")
    print("Error-type cross-table (block_error × sparse_error):")
    ct = pd.crosstab(both_wrong["block_error"], both_wrong["sparse_error"],
                     rownames=["block"], colnames=["sparse"])
    print(ct.to_string())

    # --- E1 predicted-index segment distribution ---
    seg_size = 100
    labels   = [f"{i*seg_size}-{(i+1)*seg_size-1}" for i in range(10)]

    e1_pairs = [("Block E1",  df[df["block_error"]  == "E1"]["block_pred_idx"]),
                ("Sparse E1", df[df["sparse_error"] == "E1"]["sparse_pred_idx"])]
    if any(len(s) for _, s in e1_pairs):
        print(f"\n{'='*60}")
        print("E1 predicted-index segment distribution")
        for label, series in e1_pairs:
            if len(series) == 0:
                continue
            bins = pd.cut(series, bins=range(0, 1001, seg_size),
                          right=False, labels=labels)
            print(f"\n{label} (n={len(series)}):")
            print(bins.value_counts().sort_index().to_string())

    # --- qualitative sample of both-wrong cases ---
    sample = both_wrong.sample(min(args.sample_size, len(both_wrong)), random_state=42)
    qual_path = out_dir / "qualitative_sample.txt"
    with open(qual_path, "w") as f:
        f.write(f"QUALITATIVE SAMPLE — baseline-correct, both-wrong  (n={len(sample)} of {len(both_wrong)})\n")
        f.write("=" * 70 + "\n\n")
        for _, row in sample.iterrows():
            f.write(f"ID: {row['question_id']}  [{row['category']}]  [{row['opt_dep']}]\n")
            f.write(f"Q:  {wrap(row['question'])}\n\n")
            f.write(f"GOLD  (idx={row['gold_index']})\n")
            f.write(f"  {wrap(row['gold_text'])}\n\n")
            f.write(f"BLOCK  → {row['block_error']}")
            if row["block_e2_sub"]:
                f.write(f"  [{row['block_e2_sub']}]")
            f.write(f"  (predicted idx={row['block_pred_idx']})\n")
            f.write(f"  {wrap(row['block_pred_text'])}\n\n")
            f.write(f"SPARSE → {row['sparse_error']}")
            if row["sparse_e2_sub"]:
                f.write(f"  [{row['sparse_e2_sub']}]")
            f.write(f"  (predicted idx={row['sparse_pred_idx']})\n")
            f.write(f"  {wrap(row['sparse_pred_text'])}\n")
            f.write(f"\n{SEP}\n\n")
    print(f"\nSaved qualitative sample: {qual_path}")


if __name__ == "__main__":
    main()
