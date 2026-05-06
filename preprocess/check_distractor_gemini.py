"""
Submit a Gemini batch job (File API) to judge distractor safety for every
question in an augmented dataset.

Each question's distractors are split into batches of BATCH_SIZE (default 100).
Each batch becomes one request in the Gemini batch job.

key format: {question_id}__{batch_idx:03d}
  — split on '__' to recover both components during retrieval.

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in .env.

Usage:
    python preprocess/check_distractor_gemini.py \\
        --data   data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl \\
        --model  gemini-2.5-flash-preview-05-20 \\
        --limit  50        # omit for full 1000-question run

Outputs (under out/distractor_check/<model>/<dataset_tag>/<timestamp>/):
    batch_requests.jsonl  — JSONL file uploaded to Gemini File API
    job_id.txt            — Gemini batch job name (e.g. batches/abc123)
    run_config.json       — full run configuration for reproducibility
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types

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


# Gemini does not support $defs in responseSchema, so define inline.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "flagged": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id":          {"type": "INTEGER"},
                    "reason":      {"type": "STRING"},
                    "replacement": {"type": "STRING"},
                },
                "required": ["id", "reason", "replacement"],
            },
        }
    },
    "required": ["flagged"],
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating distractors for a multiple-choice question.
Distractors are wrong answer choices that must NOT be valid or defensible answers to the question. \
They can range from highly plausible to obviously incorrect — the only strict requirement is that they are wrong.

QUESTION:
{question}

CORRECT ANSWER (for reference only):
{gold}

DISTRACTORS (labelled 0–{last_idx}):
{options_block}

TASK:
For each distractor decide whether it could be considered a correct or defensibly \
correct answer to the question above.

A distractor is UNSAFE if:
- It is factually true as an answer to this question, even if phrased differently.
- A domain expert could reasonably argue it is correct or partially correct.
- It is semantically or numerically equivalent to the correct answer.

A distractor is SAFE if a domain expert would clearly reject it as wrong.

For every UNSAFE distractor also write a replacement: a new distractor that is \
clearly wrong but plausible to someone without deep domain knowledge, \
in the same style and approximate length as the other distractors.

Respond with a JSON object only, no other text:
{{
  "flagged": [
    {{"id": <0-based index within this batch>, "reason": "<one or two sentences explaining why it is unsafe>", \
"replacement": "<new SAFE distractor text>"}}
  ]
}}

Return {{"flagged": []}} if all distractors are safe.\
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

def load_jsonl(path: Path, limit: int, domain: str | None = None) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if limit and len(records) >= limit:
                break
            rec = json.loads(line)
            if domain and rec.get("category", "").lower() != domain.lower():
                continue
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def build_requests(dataset: list) -> List[dict]:
    """Return a list of Gemini batch JSONL records (key + request)."""
    requests = []

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

            key       = f"{qid}__{b_idx:03d}"
            user_text = build_prompt(question, gold, batch)

            request = {
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema":   RESPONSE_SCHEMA,
                    "temperature":      0,
                },
            }
            requests.append({"key": key, "request": request})

    return requests


# ---------------------------------------------------------------------------
# Upload + submit
# ---------------------------------------------------------------------------

def upload_and_submit(
    requests: List[dict],
    output_dir: str,
    client: genai.Client,
    model_name: str,
) -> str:
    jsonl_path = os.path.join(output_dir, "batch_requests.jsonl")
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Uploading {len(requests)} requests …")
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(
            display_name="distractor-check-requests",
            mime_type="jsonl",
        ),
    )
    print(f"File uploaded — name: {uploaded_file.name}")

    display_name = f"distractor-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch_job = client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )

    job_name = batch_job.name
    state    = batch_job.state.name

    print(f"\nBatch job created")
    print(f"  Job name : {job_name}")
    print(f"  State    : {state}")

    with open(os.path.join(output_dir, "job_id.txt"), "w") as f:
        f.write(job_name)

    return job_name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    client = genai.Client()

    data_path   = Path(args.data)
    dataset_tag = data_path.stem
    model_slug  = args.model.replace("/", "_").replace(".", "-")
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    domain_slug = args.domain.lower().replace(" ", "_") if args.domain else None
    output_dir  = os.path.join(
        "out", "distractor_check", model_slug, dataset_tag,
        domain_slug or "all", timestamp,
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    if domain_slug:
        print(f"  Filtering by domain: {args.domain}")
    dataset = load_jsonl(data_path, args.limit, args.domain)
    print(f"  {len(dataset)} questions loaded")

    requests = build_requests(dataset)
    n_q = len(dataset)
    n_batches_per_q = len(requests) // n_q if n_q else 0
    print(f"  {len(requests)} requests ({n_q} questions × ~{n_batches_per_q} batches of {BATCH_SIZE})")

    run_config = {
        "data"       : str(data_path),
        "model"      : args.model,
        "domain"     : args.domain or "all",
        "limit"      : args.limit,
        "batch_size" : BATCH_SIZE,
        "n_questions": n_q,
        "n_requests" : len(requests),
        "timestamp"  : timestamp,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    upload_and_submit(requests, output_dir, client, args.model)

    retrieve_cmd = (
        f"python3 preprocess/retrieve_distractor_check_gemini.py \\\n"
        f"    --output-dir {output_dir} \\\n"
        f"    --data {args.data}"
    )
    with open(os.path.join(output_dir, "retrieve_cmd.txt"), "w") as f:
        f.write(retrieve_cmd + "\n")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"\nRetrieve results with:")
    print(f"  {retrieve_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="Path to augmented dataset JSONL (block or sparse)")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Gemini model name (default: gemini-3-flash-preview)")
    parser.add_argument("--domain", default=None,
                        help="Filter by category (e.g. 'health', 'law'); case-insensitive. Omit for all domains.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of questions (0 = all)")
    args = parser.parse_args()
    main(args)
