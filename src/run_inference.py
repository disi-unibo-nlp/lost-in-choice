"""
Submit a Together AI batch inference job on one augmented MMLU-Pro configuration.

Usage:
    python src/run_inference.py \\
        --window 0.5 \\
        --mode sparse \\
        --model-name "openai/gpt-oss-120b" \\
        --limit 10          # omit for full run

Dataset path resolved automatically from --window and --mode:
    data/bench/mmlu-pro-augmented-1000/window_{w:.2f}/{mode}/

Structured decoding:
    Every request uses response_format = json_schema with schema
        {"answer_index": <integer 0 … N_OPTIONS-1>}
    so the model is *forced* to emit a valid integer index.
    For reasoning models (gpt-oss-*) temperature is set to 0 and
    reasoning_effort is forwarded; for all other models a system
    prompt is injected and max_tokens is set to 4096.

Outputs (under out/completions/<model>/<dataset_tag>/<timestamp>/):
    batch_requests.jsonl   — the JSONL file uploaded to Together
    job_id.txt             — Together batch job ID
    run_config.json        — full run configuration for reproducibility
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple

from datasets import load_from_disk
from dotenv import load_dotenv
from together import Together

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_WINDOWS = [0.0, 0.25, 0.5, 0.75, 1.0]
VALID_MODES = ["block", "sparse"]

# JSON schema for the structured answer — a single integer index
def answer_json_schema(n_options: int, is_instruct_model: bool = False) -> dict:
    properties: dict = {
        "answer_index": {
            "type": "integer",
            "minimum": 0,
            "maximum": n_options - 1,
            "description": f"The index (0–{n_options - 1}) of the correct option.",
        }
    }
    required = ["answer_index"]
    if is_instruct_model:
        properties["reasoning"] = {
            "type": "string",
            "description": "Step-by-step reasoning before selecting the answer.",
        }
        required = ["reasoning", "answer_index"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mcq_answer",
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


# SYSTEM_PROMPT = (
#     "You are an expert at answering multiple-choice questions. "
#     "You will be given a question followed by a numbered list of options. "
#     "Identify the single correct option and return its index as JSON."
# )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(item: dict, n_options: int, options_key: str = "new_options", is_instruct_model: bool = False) -> Tuple[str, str]:
    """Return (custom_id, user_message_text) for one dataset row."""
    qid = str(item["question_id"])
    question = item["question"]
    options: List[str] = item[options_key]

    lines = [f"Question: {question}", "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"({i}) {opt}")

    if not is_instruct_model:
        lines += [
            "",
            f"Select the single correct option. "
            f"Respond with JSON: {{\"answer_index\": <integer from 0 to {n_options - 1}>}}",
        ]
    else:
        lines += [
            "",
            f"Select the single correct option. "
            f"Respond with JSON: {{\"reasoning\": <step by step reasoning>, \"answer_index\": <integer from 0 to {n_options - 1}>}}",
        ]

    return qid, "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch request builder
# ---------------------------------------------------------------------------

def build_requests(
    dataset,
    model_name: str,
    n_options: int,
    reasoning_effort: str,
    options_key: str = "new_options",
    is_instruct_model: bool = False,
) -> List[dict]:
    schema = answer_json_schema(n_options, is_instruct_model=is_instruct_model)
    requests = []

    for item in dataset:
        custom_id, user_text = format_prompt(item, n_options, options_key, is_instruct_model=is_instruct_model)

        body: dict = {
            "model": model_name,
            "response_format": schema,
        }

        body["messages"] = [{"role": "user", "content": user_text}]
        if "gpt-oss" in model_name.lower():
            body["reasoning_effort"] = reasoning_effort
        body["max_tokens"] = 32000 

        requests.append({"custom_id": custom_id, "body": body})

    return requests


# ---------------------------------------------------------------------------
# Together API helpers
# ---------------------------------------------------------------------------

def upload_and_submit(
    requests: List[dict],
    output_dir: str,
    client: Together,
) -> str:
    jsonl_path = os.path.join(output_dir, "batch_requests.jsonl")
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Uploading {len(requests)} requests from: {jsonl_path}")
    file_resp = client.files.upload(file=jsonl_path, purpose="batch-api", check=False)
    print(f"File uploaded — id: {file_resp.id}")

    batch = client.batches.create(
        input_file_id=file_resp.id,
        endpoint="/v1/chat/completions",
    )
    job_id = batch.job.id
    batch_stat = client.batches.retrieve(job_id)

    print(f"\nBatch job created")
    print(f"  Job ID : {job_id}")
    print(f"  Status : {batch_stat.status}")

    with open(os.path.join(output_dir, "job_id.txt"), "w") as f:
        f.write(job_id)

    return job_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    # --- Resolve dataset path and field names ---
    if args.baseline:
        dataset_path = "data/bench/mmlu-pro-augmented-10"
        options_key = "options"
        answer_index_key = "answer_index"
        window_str = "baseline"
    else:
        bench_root = f"data/bench/mmlu-pro-augmented-clean-{args.n_options}"
        window_str = f"{args.window:.2f}"
        dataset_path = os.path.join(bench_root, f"seed_{args.seed}", f"window_{window_str}", args.mode)
        options_key = "new_options"
        answer_index_key = "new_answer_index"

    if not os.path.isdir(dataset_path):
        print(
            f"ERROR: dataset not found at '{dataset_path}'.\n"
            f"Run {'sample_final_dataset.py' if args.baseline else 'augment_final_dataset.py'} first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading dataset: {dataset_path}")
    ds = load_from_disk(dataset_path)

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    n_options = len(ds[0][options_key])
    if args.baseline:
        print(f"  {len(ds)} questions  |  {n_options} options each  [baseline]")
    else:
        print(f"  {len(ds)} questions  |  {n_options} options each")
        print(f"  seed={args.seed}  window={args.window}  mode={args.mode}")

    # Sanity-check gold positions
    indices = [row[answer_index_key] for row in ds]
    print(f"  Gold index — min={min(indices)}  max={max(indices)}  mean={sum(indices)/len(indices):.1f}")

    # --- Output dir ---
    model_slug = args.model_name.split("/")[-1]
    if "gpt-oss" in args.model_name.lower():
        model_slug = f"{model_slug}-{args.reasoning_effort}"
    if args.baseline:
        dataset_tag = "mmlu-pro-augmented-10__baseline"
    else:
        dataset_tag = f"mmlu-pro-augmented-{args.n_options}__seed_{args.seed}__window_{window_str}__{args.mode}"
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, model_slug, dataset_tag, now_str)
    os.makedirs(output_dir, exist_ok=True)

    # --- Save run config for reproducibility ---
    config = {
        "model_name": args.model_name,
        "baseline": args.baseline,
        "seed": args.seed if not args.baseline else None,
        "window": args.window if not args.baseline else None,
        "mode": args.mode if not args.baseline else None,
        "n_options": n_options,
        "n_questions": len(ds),
        "reasoning_effort": args.reasoning_effort,
        "limit": args.limit,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "timestamp": now_str,
        "is_instruct_model": args.is_instruct_model,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nRun config saved to: {output_dir}/run_config.json")

    # Print a sample prompt for inspection
    sample_id, sample_prompt = format_prompt(ds[0], n_options, options_key, args.is_instruct_model)
    print(f"\n--- Sample prompt (question_id={sample_id}) ---")
    sample_lines = sample_prompt.split("\n")
    preview = sample_lines[:6] + ["  ...", f"  ({n_options-1}) {ds[0][options_key][-1]}"] + sample_lines[-2:]
    print("\n".join(preview))
    print("--- end sample ---\n")

    # --- Build requests ---
    print("Building batch requests...")
    requests = build_requests(
        dataset=ds,
        model_name=args.model_name,
        n_options=n_options,
        reasoning_effort=args.reasoning_effort,
        options_key=options_key,
        is_instruct_model=args.is_instruct_model,
    )

    # --- Upload and submit ---
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in environment. Check your .env file.")

    client = Together(api_key=TOGETHER_API_KEY)

    job_id = upload_and_submit(requests, output_dir, client)

    retrieve_cmd = (
        f"python3 src/retrieve_inference.py "
        f"--output-dir {output_dir} "
        f"--job-id {job_id}"
    )
    with open(os.path.join(output_dir, "retrieve_command.txt"), "w") as f:
        f.write(retrieve_cmd + "\n")

    print(f"\nDone. Retrieve results with:")
    print(f"  {retrieve_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a Together AI batch inference job on one augmented MMLU-Pro configuration."
    )
    parser.add_argument(
        "--window",
        type=float,
        default=0.5,
        choices=VALID_WINDOWS,
        metavar="{" + ",".join(str(w) for w in VALID_WINDOWS) + "}",
        help="Sliding window position for the gold answer (0.0=start … 1.0=end).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sparse",
        choices=VALID_MODES,
        help="Placement mode: 'block' (original 10 contiguous) or 'sparse' (spread across segments).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/gpt-oss-120b",
        help="Together AI model path (e.g. 'openai/gpt-oss-120b', 'meta-llama/Llama-3.3-70B-Instruct-Turbo').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions",
        help="Base directory for output files.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort level (only used for gpt-oss-* reasoning models).",
    )
    parser.add_argument(
        "--n-options",
        type=int,
        default=1000,
        help="Number of options in the augmented dataset (e.g. 100, 500, 1000). Used to locate data/bench/mmlu-pro-augmented-clean-<n>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Distractor sampling seed — selects the seed_<N>/ subfolder under the benchmark root.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions submitted (useful for quick smoke tests).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run on the original 10-option baseline (data/bench/mmlu-pro-augmented-10). "
             "Ignores --window, --mode, --seed, and --n-options.",
    )

    args = parser.parse_args()
    args.is_instruct_model = "instruct" in args.model_name.lower()
    main(args)
