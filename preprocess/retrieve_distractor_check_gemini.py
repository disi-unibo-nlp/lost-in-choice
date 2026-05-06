"""
Retrieve results from a completed Gemini batch job and reconstruct per-question
flagged distractor lists with suggested replacements.

key format written by check_distractor_gemini.py:
    {question_id}__{batch_idx:03d}

Usage:
    python preprocess/retrieve_distractor_check_gemini.py \\
        --output-dir out/distractor_check/gemini-2-5-flash-preview/block/2026-04-25_12-00-00 \\
        --data       data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl

    # job name is read from output-dir/job_id.txt automatically if present
    python preprocess/retrieve_distractor_check_gemini.py \\
        --output-dir out/distractor_check/gemini-2-5-flash-preview/block/2026-04-25_12-00-00 \\
        --data       data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl \\
        --job-id     batches/abc123

Outputs (written into --output-dir):
    raw_completions.jsonl      — raw JSONL as downloaded from Gemini
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
from google import genai

load_dotenv()

BATCH_SIZE = 100

# Terminal states — stop polling when reached
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


# ---------------------------------------------------------------------------
# Schema (must match check_distractor_gemini.py)
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

def retrieve_raw(job_name: str, output_dir: str, client: genai.Client) -> str:
    print(f"Checking batch status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)
    state = batch_job.state.name
    print(f"  State: {state}")

    if state not in TERMINAL_STATES:
        print(
            f"Batch not yet in a terminal state ({state}). Re-run once the job finishes.",
            file=sys.stderr,
        )
        sys.exit(1)

    if state != "JOB_STATE_SUCCEEDED":
        print(f"Batch ended with state {state}. Cannot retrieve results.", file=sys.stderr)
        sys.exit(1)

    result_file_name = batch_job.dest.file_name
    print(f"Downloading output file: {result_file_name}")
    file_content = client.files.download(file=result_file_name)

    raw_path = os.path.join(output_dir, "raw_completions.jsonl")
    with open(raw_path, "wb") as f:
        f.write(file_content)
    print(f"Download complete → {raw_path}")
    return raw_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_response(line: dict) -> Optional[JudgeResponse]:
    """Extract JudgeResponse from one Gemini batch output line."""
    try:
        text = line["response"]["candidates"][0]["content"]["parts"][0]["text"]
        return JudgeResponse.model_validate_json(text)
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

    by_question: dict = defaultdict(dict)  # qid → {batch_idx: JudgeResponse}
    parse_failures = 0

    for item in completions:
        key = item.get("key", "")

        if "__" not in key:
            print(f"  [WARN] unexpected key format: {key!r}")
            continue

        qid, b_idx_str = key.rsplit("__", 1)
        b_idx = int(b_idx_str)

        # Skip lines that represent errors (have "status" instead of "response")
        if "status" in item and "response" not in item:
            print(f"  [WARN] error response for key={key!r}: {item.get('status')}")
            parse_failures += 1
            continue

        result = parse_response(item)
        if result is None:
            parse_failures += 1
            print(f"  [WARN] parse failure for key={key!r}")
            continue

        by_question[qid][b_idx] = result

    print(f"  {len(by_question)} questions with responses  |  {parse_failures} parse failures")

    flagged_records = []
    total_checked   = 0
    total_flagged   = 0

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

    flagged_path = os.path.join(output_dir, "flagged_distractors.jsonl")
    with open(flagged_path, "w") as f:
        for rec in flagged_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved: {flagged_path}")

    n_q_with_flags = sum(1 for r in flagged_records if r["n_flagged"] > 0)
    summary = {
        "n_questions"           : len(flagged_records),
        "n_questions_with_flags": n_q_with_flags,
        "n_distractors_checked" : total_checked,
        "n_flagged"             : total_flagged,
        "pct_flagged"           : round(100 * total_flagged / total_checked, 2) if total_checked else 0,
        "parse_failures"        : parse_failures,
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
    client     = genai.Client()
    output_dir = args.output_dir

    job_name = args.job_id
    if not job_name:
        id_path = os.path.join(output_dir, "job_id.txt")
        if os.path.exists(id_path):
            job_name = open(id_path).read().strip()
        else:
            print("No --job-id provided and job_id.txt not found.", file=sys.stderr)
            sys.exit(1)

    raw_path = retrieve_raw(job_name, output_dir, client)

    print(f"\nLoading dataset from: {args.data}")
    dataset = load_jsonl(Path(args.data))
    print(f"  {len(dataset)} questions loaded")

    reconstruct(raw_path, dataset, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Directory created by check_distractor_gemini.py")
    parser.add_argument("--data", required=True,
                        help="Same JSONL passed to check_distractor_gemini.py")
    parser.add_argument("--job-id", default="",
                        help="Gemini batch job name, e.g. batches/abc123 "
                             "(auto-read from job_id.txt if omitted)")
    args = parser.parse_args()
    main(args)
