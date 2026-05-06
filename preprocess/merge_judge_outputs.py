"""
Merge two flagged_distractors.jsonl files, substituting one judge's results
for a specific category with another judge's.

Usage:
    python preprocess/merge_judge_outputs.py \\
        --base        out/distractor_check/openai_gpt-oss-120b/block/2026-04-24_15-04-12/flagged_distractors.jsonl \\
        --override    out/distractor_check/gemini-3-flash-preview/block/math/2026-04-25_01-44-10/flagged_distractors.jsonl \\
        --category    math \\
        --output      out/distractor_check/merged/flagged_distractors.jsonl

For every record whose category matches --category, the override file's record
replaces the base file's record. All other records are taken from the base file.
The output preserves the base file's question ordering.
Also writes a summary.json next to the output.
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main(args):
    base_records     = load_jsonl(Path(args.base))
    override_records = load_jsonl(Path(args.override))
    category         = args.category.lower().strip()

    # Index override by question_id, restricted to the target category
    override_by_qid = {
        str(r["question_id"]): r
        for r in override_records
        if r.get("category", "").lower() == category and r.get("n_distractors_checked", 0) > 0
    }
    print(f"Base records       : {len(base_records)}")
    print(f"Override records   : {len(override_records)}")
    print(f"Override with data : {len(override_by_qid)} (category={category!r})")

    merged = []
    substituted = 0
    for rec in base_records:
        qid = str(rec["question_id"])
        if rec.get("category", "").lower() == category and qid in override_by_qid:
            merged.append(override_by_qid[qid])
            substituted += 1
        else:
            merged.append(rec)

    print(f"Substituted        : {substituted} records")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")
    print(f"Written            : {out_path}")

    total_checked = sum(r["n_distractors_checked"] for r in merged)
    total_flagged = sum(r["n_flagged"] for r in merged)
    n_with_flags  = sum(1 for r in merged if r["n_flagged"] > 0)
    summary = {
        "n_questions"           : len(merged),
        "n_questions_with_flags": n_with_flags,
        "n_distractors_checked" : total_checked,
        "n_flagged"             : total_flagged,
        "pct_flagged"           : round(100 * total_flagged / total_checked, 2) if total_checked else 0,
        "base"                  : args.base,
        "override"              : args.override,
        "override_category"     : category,
    }
    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Written            : {summary_path}")
    print(f"\nSUMMARY")
    print(f"  Questions           : {summary['n_questions']}")
    print(f"  With ≥1 flag        : {summary['n_questions_with_flags']}")
    print(f"  Distractors checked : {summary['n_distractors_checked']}")
    print(f"  Flagged             : {summary['n_flagged']}  ({summary['pct_flagged']}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",     required=True, help="Base flagged_distractors.jsonl")
    parser.add_argument("--override", required=True, help="Override flagged_distractors.jsonl")
    parser.add_argument("--category", required=True, help="Category to substitute (e.g. math)")
    parser.add_argument("--output",   required=True, help="Output path for merged JSONL")
    args = parser.parse_args()
    main(args)
