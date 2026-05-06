"""
Submit a Together AI batch job to judge distractor safety for every question
in an augmented dataset.

Each question's 990 distractors are split into batches of BATCH_SIZE (default
100).  Each batch becomes one request in the Together batch job.

custom_id format: {question_id}__{batch_idx:03d}
  — split on '__' to recover both components during retrieval.

Usage:
    python preprocess/check_distractors.py \\
        --data   data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl \\
        --model  openai/gpt-oss-120b \\
        --limit  50        # omit for full 1000-question run

Outputs (under out/distractor_check/<model>/<dataset_tag>/<timestamp>/):
    batch_requests.jsonl  — JSONL file uploaded to Together
    job_id.txt            — Together batch job ID
    run_config.json       — full run configuration for reproducibility
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from together import Together

load_dotenv()

BATCH_SIZE = 100  # distractors per request


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class FlaggedOption(BaseModel):
    id: int           # 0-based index within the batch
    reason: str
    replacement: str  # safe replacement written by the model


class JudgeResponse(BaseModel):
    flagged: List[FlaggedOption]


def judge_json_schema() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "distractor_judge",
            "schema": JudgeResponse.model_json_schema(),
            "strict": True,
        },
    }


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating answer options for a multiple-choice question.

QUESTION:
{question}

CORRECT ANSWER (for reference only):
{gold}

CANDIDATE OPTIONS (labelled 0–{last_idx}):
{options_block}

TASK:
For each option decide whether it could be considered a correct or defensibly \
correct answer to the question above.

An option is UNSAFE if:
- It is factually true as an answer to this question, even if phrased differently.
- A domain expert could reasonably argue it is correct or partially correct.
- It is semantically or numerically equivalent to the correct answer.

An option is SAFE if a domain expert would clearly reject it as wrong.

For every UNSAFE option also write a replacement: a new answer that is \
clearly wrong but plausible to someone without deep domain knowledge, \
in the same style and approximate length as the other options.

Respond with a JSON object only, no other text:
{{
  "flagged": [
    {{"id": <0-based index within this batch>, "reason": "<one or two sentences explaining why it is unsafe>", \
"replacement": "<new SAFE option text>"}}
  ]
}}

Return {{"flagged": []}} if all options are safe.\
"""


def format_options_block(options: list) -> str:
    return "\n".join(f"({i}) {opt}" for i, opt in enumerate(options))


def build_prompt(question: str, gold: str, batch: list) -> str:
    return JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        last_idx=len(batch) - 1,
        options_block=format_options_block(batch),
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, limit: int) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if limit and len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def build_requests(dataset: list, model_name: str) -> List[dict]:
    requests = []
    schema = judge_json_schema()

    for ex in dataset:
        qid          = str(ex["question_id"])
        question     = ex["question"]
        orig_options = ex["options"]
        gold         = orig_options[ex["answer_index"]]
        orig_set     = set(orig_options)

        distractors = [o for o in ex["new_options"] if o not in orig_set]

        for b_idx, start in enumerate(range(0, len(distractors), BATCH_SIZE)):
            batch = distractors[start : start + BATCH_SIZE]
            if not batch:
                break

            custom_id = f"{qid}__{b_idx:03d}"
            user_text = build_prompt(question, gold, batch)

            body = {
                "model"          : model_name,
                "messages"       : [{"role": "user", "content": user_text}],
                "response_format": schema,
                "max_tokens"     : 4096,
                "temperature"    : 0,
            }
            if "gpt-oss" in model_name.lower():
                body["reasoning_effort"] = "low"

            requests.append({"custom_id": custom_id, "body": body})

    return requests


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

def upload_and_submit(requests: List[dict], output_dir: str, client: Together) -> str:
    jsonl_path = os.path.join(output_dir, "batch_requests.jsonl")
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Uploading {len(requests)} requests …")
    file_resp = client.files.upload(file=jsonl_path, purpose="batch-api", check=False)
    print(f"File uploaded — id: {file_resp.id}")

    batch = client.batches.create(
        input_file_id=file_resp.id,
        endpoint="/v1/chat/completions",
    )
    job_id = batch.job.id
    status = client.batches.retrieve(job_id).status

    print(f"\nBatch job created")
    print(f"  Job ID : {job_id}")
    print(f"  Status : {status}")

    with open(os.path.join(output_dir, "job_id.txt"), "w") as f:
        f.write(job_id)

    return job_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    client = Together()

    data_path   = Path(args.data)
    dataset_tag = data_path.stem           # e.g. "block"
    model_slug  = args.model.replace("/", "_")
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir  = os.path.join(
        "out", "distractor_check", model_slug, dataset_tag, timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    dataset = load_jsonl(data_path, args.limit)
    print(f"  {len(dataset)} questions loaded")

    requests = build_requests(dataset, args.model)
    n_q = len(dataset)
    n_batches_per_q = len(requests) // n_q if n_q else 0
    print(f"  {len(requests)} requests ({n_q} questions × ~{n_batches_per_q} batches of {BATCH_SIZE})")

    run_config = {
        "data"      : str(data_path),
        "model"     : args.model,
        "limit"     : args.limit,
        "batch_size": BATCH_SIZE,
        "n_questions": n_q,
        "n_requests": len(requests),
        "timestamp" : timestamp,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    job_id = upload_and_submit(requests, output_dir, client)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"\nRetrieve results with:")
    print(f"  python preprocess/retrieve_distractor_check.py \\")
    print(f"    --output-dir {output_dir} \\")
    print(f"    --data {args.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to augmented dataset JSONL (block or sparse)")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of questions (0 = all)")
    args = parser.parse_args()
    main(args)
