"""
Retrieve results from a completed Together AI distractor-check batch job and
reconstruct per-question flagged distractor lists with suggested replacements.

custom_id format written by check_distractors.py:
    {question_id}__{batch_idx:03d}

Usage:
    python preprocess/retrieve_distractor_check.py \\
        --output-dir out/distractor_check/openai_gpt-oss-120b/block/2026-04-24_12-00-00 \\
        --data       data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl

    # job-id is read from output-dir/job_id.txt automatically if present
    python preprocess/retrieve_distractor_check.py \\
        --output-dir out/distractor_check/openai_gpt-oss-120b/block/2026-04-24_12-00-00 \\
        --data       data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl \\
        --job-id     <BATCH_JOB_ID>

Outputs (written into --output-dir):
    raw_completions.jsonl      — raw API response lines as downloaded
    flagged_distractors.jsonl  — one record per question:
        {
          "question_id": "...",
          "question": "...",
          "gold": "...",
          "category": "...",
          "n_distractors_checked": 990,
          "n_flagged": 3,
          "flagged": [
            {
              "batch_idx": 2,
              "option_idx_in_batch": 7,
              "global_distractor_idx": 207,
              "original_text": "...",
              "reason": "...",
              "replacement": "..."
            }
          ]
        }
    summary.json               — aggregate stats
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from together import Together

load_dotenv()

BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Schema (must match check_distractors.py)
# ---------------------------------------------------------------------------

class FlaggedOption(BaseModel):
    id: int
    reason: str
    replacement: str


class JudgeResponse(BaseModel):
    flagged: List[FlaggedOption]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_raw(job_id: str, output_dir: str, client: Together) -> str:
    print(f"Checking batch status for job: {job_id}")
    batch = client.batches.retrieve(job_id)
    print(f"  Status: {batch.status}")

    if batch.status != "COMPLETED":
        print("Batch not yet completed. Re-run once the job finishes.", file=sys.stderr)
        sys.exit(1)

    raw_path = os.path.join(output_dir, "raw_completions.jsonl")
    print(f"Downloading results to: {raw_path}")
    with client.files.with_streaming_response.content(id=batch.output_file_id) as resp:
        with open(raw_path, "wb") as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
    print("Download complete.")
    return raw_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_response(response_body: dict) -> Optional[JudgeResponse]:
    try:
        content = response_body["choices"][0]["message"]["content"]
        return JudgeResponse.model_validate_json(content)
    except Exception:
        return None


def load_jsonl(path: Path, limit: int = 0) -> dict:
    """Load JSONL indexed by question_id."""
    records = {}
    with open(path) as f:
        for line in f:
            if limit and len(records) >= limit:
                break
            d = json.loads(line)
            records[str(d["question_id"])] = d
    return records


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(raw_path: str, dataset: dict, output_dir: str):
    print("\nParsing completions …")
    with open(raw_path) as f:
        completions = [json.loads(line) for line in f if line.strip()]
    print(f"  {len(completions)} completion lines loaded")

    # Group batch responses by question_id
    by_question: dict = defaultdict(dict)  # qid → {batch_idx: JudgeResponse}
    parse_failures = 0

    for item in completions:
        custom_id    = item.get("custom_id", "")
        response_body = item.get("response", {}).get("body", {})

        if "__" not in custom_id:
            print(f"  [WARN] unexpected custom_id format: {custom_id!r}")
            continue

        qid, b_idx_str = custom_id.rsplit("__", 1)
        b_idx = int(b_idx_str)

        result = parse_response(response_body)
        if result is None:
            parse_failures += 1
            print(f"  [WARN] parse failure for custom_id={custom_id!r}")
            continue

        by_question[qid][b_idx] = result

    print(f"  {len(by_question)} questions with responses  |  {parse_failures} parse failures")

    # Build per-question flagged records
    flagged_records = []
    total_checked = 0
    total_flagged = 0

    for qid, ex in dataset.items():
        orig_options = ex["options"]
        gold         = orig_options[ex["answer_index"]]
        orig_set     = set(orig_options)
        distractors  = [o for o in ex["new_options"] if o not in orig_set]

        batch_results = by_question.get(qid, {})
        flagged_items = []

        for b_idx, result in sorted(batch_results.items()):
            batch_start = b_idx * BATCH_SIZE
            batch       = distractors[batch_start : batch_start + BATCH_SIZE]

            for item in result.flagged:
                if item.id >= len(batch):
                    print(f"  [WARN] qid={qid} batch={b_idx}: id={item.id} out of range "
                          f"(batch size={len(batch)})")
                    continue
                global_idx = batch_start + item.id
                flagged_items.append({
                    "batch_idx"            : b_idx,
                    "option_idx_in_batch"  : item.id,
                    "global_distractor_idx": global_idx,
                    "original_text"        : batch[item.id],
                    "reason"               : item.reason,
                    "replacement"          : item.replacement,
                })

        n_checked = sum(
            len(distractors[b * BATCH_SIZE : (b + 1) * BATCH_SIZE])
            for b in batch_results
        )
        total_checked += n_checked
        total_flagged += len(flagged_items)

        flagged_records.append({
            "question_id"           : qid,
            "question"              : ex["question"],
            "gold"                  : gold,
            "category"              : ex.get("category", ""),
            "opt_dep"               : ex.get("option_dependent_vs_independent", ""),
            "n_distractors_checked" : n_checked,
            "n_flagged"             : len(flagged_items),
            "flagged"               : flagged_items,
        })

    # Save flagged_distractors.jsonl
    flagged_path = os.path.join(output_dir, "flagged_distractors.jsonl")
    with open(flagged_path, "w") as f:
        for rec in flagged_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved: {flagged_path}")

    # Summary
    n_q_with_flags = sum(1 for r in flagged_records if r["n_flagged"] > 0)
    summary = {
        "n_questions"          : len(flagged_records),
        "n_questions_with_flags": n_q_with_flags,
        "n_distractors_checked": total_checked,
        "n_flagged"            : total_flagged,
        "pct_flagged"          : round(100 * total_flagged / total_checked, 2) if total_checked else 0,
        "parse_failures"       : parse_failures,
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {summary_path}")
    print(f"\nSUMMARY")
    print(f"  Questions           : {summary['n_questions']}")
    print(f"  With ≥1 flag        : {summary['n_questions_with_flags']}")
    print(f"  Distractors checked : {summary['n_distractors_checked']}")
    print(f"  Flagged             : {summary['n_flagged']}  ({summary['pct_flagged']}%)")
    print(f"  Parse failures      : {summary['parse_failures']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    client     = Together()
    output_dir = args.output_dir

    # resolve job_id
    job_id = args.job_id
    if not job_id:
        id_path = os.path.join(output_dir, "job_id.txt")
        if os.path.exists(id_path):
            job_id = open(id_path).read().strip()
        else:
            print("No --job-id provided and job_id.txt not found.", file=sys.stderr)
            sys.exit(1)

    raw_path = retrieve_raw(job_id, output_dir, client)

    print(f"\nLoading dataset from: {args.data}")
    dataset = load_jsonl(Path(args.data))
    print(f"  {len(dataset)} questions loaded")

    reconstruct(raw_path, dataset, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Directory created by check_distractors.py")
    parser.add_argument("--data", required=True,
                        help="Same JSONL passed to check_distractors.py")
    parser.add_argument("--job-id", default="",
                        help="Together batch job ID (auto-read from job_id.txt if omitted)")
    args = parser.parse_args()
    main(args)
