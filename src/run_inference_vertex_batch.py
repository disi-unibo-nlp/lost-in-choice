"""
Submit a Vertex AI (Gemini) batch inference job on one augmented MMLU-Pro configuration.

Usage:
    python src/run_inference_vertex_batch.py \\
        --window 0.5 \\
        --mode sparse \\
        --vertex-model "gemini-2.5-flash" \\
        --project my-gcp-project \\
        --bucket my-gcs-bucket \\
        --limit 10          # omit for full run

GCS credentials are picked up automatically from the environment
(gcloud auth application-default login, or GOOGLE_APPLICATION_CREDENTIALS).

The batch JSONL is built locally, uploaded to GCS, and submitted via the
Vertex AI batch prediction API — no manual upload required.

Outputs (under out/completions/<model>/<dataset_tag>/<timestamp>/):
    batch_requests.jsonl   — JSONL uploaded to GCS (local copy)
    question_order.json    — ordered list of question IDs (used as fallback
                             if Vertex AI does not echo custom_id in output)
    job_name.txt           — Vertex AI batch job resource name
    run_config.json        — full run configuration for reproducibility
    retrieve_command.txt   — ready-to-run retrieval command
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Tuple

import fsspec
from datasets import load_from_disk
from dotenv import load_dotenv
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions

load_dotenv()

VALID_WINDOWS = [0.0, 0.25, 0.5, 0.75, 1.0]
VALID_MODES = ["block", "sparse"]


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def answer_response_schema(n_options: int) -> dict:
    return {
        "type": "OBJECT",
        "properties": {
            "answer_index": {
                "type": "INTEGER",
                "description": f"The index (0–{n_options - 1}) of the correct option.",
            }
        },
        "required": ["answer_index"],
    }


# ---------------------------------------------------------------------------
# Prompt formatting (same structure as run_inference.py)
# ---------------------------------------------------------------------------

def format_prompt(item: dict, n_options: int, options_key: str) -> Tuple[str, str]:
    """Return (custom_id, user_message_text) for one dataset row."""
    qid = str(item["question_id"])
    question = item["question"]
    options: List[str] = item[options_key]

    lines = [f"Question: {question}", "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"({i}) {opt}")
    lines += [
        "",
        f"Select the single correct option. "
        f"Respond with JSON: {{\"answer_index\": <integer from 0 to {n_options - 1}>}}",
    ]
    return qid, "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch request builder (Gemini format)
# ---------------------------------------------------------------------------

VALID_THINKING_LEVELS = ["MINIMAL", "LOW", "MEDIUM", "HIGH"]


def build_requests(
    dataset,
    n_options: int,
    options_key: str,
    thinking_level: str,
) -> List[dict]:
    schema = answer_response_schema(n_options)
    generation_config: dict = {
        "responseMimeType": "application/json",
        "responseSchema": schema,
        # Gemini 3 uses thinkingLevel (not thinkingBudget).
        # includeThoughts=True returns the reasoning trace in the response parts.
        # temperature and maxOutputTokens are omitted: temperature is fixed for
        # thinking models and capping maxOutputTokens would truncate reasoning traces.
        "thinkingConfig": {
            "thinkingLevel": thinking_level,
            "includeThoughts": True,
        },
    }

    requests = []
    for item in dataset:
        qid, user_text = format_prompt(item, n_options, options_key)
        requests.append({
            "custom_id": qid,
            "request": {
                "contents": [{"role": "user", "parts": [{"text": user_text}]}],
                "generationConfig": generation_config,
            },
        })
    return requests


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------

def upload_to_gcs(local_path: str, gcs_uri: str) -> None:
    """Upload a local file to GCS using fsspec/gcsfs."""
    fs = fsspec.filesystem("gcs")
    gcs_path = gcs_uri.removeprefix("gs://")
    fs.put(local_path, gcs_path)
    print(f"Uploaded → {gcs_uri}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    # --- Resolve dataset path ---
    if args.baseline:
        dataset_path = "data/bench/mmlu-pro-augmented-10"
        options_key = "options"
        answer_index_key = "answer_index"
        window_str = "baseline"
    else:
        bench_root = f"data/bench/mmlu-pro-augmented-clean-{args.n_options}"
        window_str = f"{args.window:.2f}"
        dataset_path = os.path.join(
            bench_root, f"seed_{args.seed}", f"window_{window_str}", args.mode
        )
        options_key = "new_options"
        answer_index_key = "new_answer_index"

    if not os.path.isdir(dataset_path):
        print(f"ERROR: dataset not found at '{dataset_path}'.", file=sys.stderr)
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

    indices = [row[answer_index_key] for row in ds]
    print(f"  Gold index — min={min(indices)}  max={max(indices)}  mean={sum(indices)/len(indices):.1f}")

    # --- Output dirs ---
    model_slug = args.vertex_model.replace("/", "-").replace(".", "-")
    if args.baseline:
        dataset_tag = "mmlu-pro-augmented-10__baseline"
    else:
        dataset_tag = (
            f"mmlu-pro-augmented-{args.n_options}"
            f"__seed_{args.seed}"
            f"__window_{window_str}"
            f"__{args.mode}"
        )
    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, model_slug, dataset_tag, now_str)
    os.makedirs(output_dir, exist_ok=True)

    gcs_input_uri = (
        f"gs://{args.bucket}/inference/{model_slug}/{dataset_tag}/{now_str}/batch_requests.jsonl"
    )
    gcs_output_base = (
        f"gs://{args.bucket}/inference/{model_slug}/{dataset_tag}/{now_str}/output"
    )

    # Print sample prompt for inspection
    sample_id, sample_prompt = format_prompt(ds[0], n_options, options_key)
    sample_lines = sample_prompt.split("\n")
    preview = sample_lines[:6] + ["  ...", f"  ({n_options-1}) {ds[0][options_key][-1]}"] + sample_lines[-2:]
    print(f"\n--- Sample prompt (question_id={sample_id}) ---")
    print("\n".join(preview))
    print("--- end sample ---\n")

    # --- Build and save requests ---
    print("Building batch requests...")
    requests = build_requests(ds, n_options, options_key, args.thinking_level)

    local_jsonl = os.path.join(output_dir, "batch_requests.jsonl")
    with open(local_jsonl, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"Saved locally: {local_jsonl}  ({len(requests)} requests)")

    # Save ordered question IDs for fallback matching in retrieval
    qid_order = [req["custom_id"] for req in requests]
    with open(os.path.join(output_dir, "question_order.json"), "w") as f:
        json.dump(qid_order, f)

    # --- Upload to GCS ---
    upload_to_gcs(local_jsonl, gcs_input_uri)

    # --- Submit Vertex AI batch job ---
    project = args.project or os.getenv("VERTEX_PROJECT")
    if not project:
        print("ERROR: GCP project not set. Use --project or VERTEX_PROJECT env var.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(
        vertexai=True,
        project=project,
        location=args.location,
        http_options=HttpOptions(api_version="v1"),
    )

    print(f"\nSubmitting Vertex AI batch job...")
    print(f"  Model  : {args.vertex_model}")
    print(f"  Input  : {gcs_input_uri}")
    print(f"  Output : {gcs_output_base}")

    batch_job = client.batches.create(
        model=args.vertex_model,
        src=gcs_input_uri,
        config=CreateBatchJobConfig(dest=gcs_output_base),
    )
    job_name = batch_job.name

    # Refresh once to get actual dest URI
    batch_job = client.batches.get(name=job_name)
    actual_output_uri = (
        batch_job.dest.gcs_uri
        if hasattr(batch_job, "dest") and batch_job.dest and batch_job.dest.gcs_uri
        else gcs_output_base
    )

    print(f"  Job name : {job_name}")
    print(f"  State    : {batch_job.state}")
    print(f"  Output   : {actual_output_uri}")

    with open(os.path.join(output_dir, "job_name.txt"), "w") as f:
        f.write(job_name + "\n")

    # --- Save run config ---
    config = {
        "model_name": args.vertex_model,
        "vertex_project": project,
        "vertex_location": args.location,
        "gcs_bucket": args.bucket,
        "gcs_input_uri": gcs_input_uri,
        "gcs_output_uri": actual_output_uri,
        "job_name": job_name,
        "baseline": args.baseline,
        "seed": args.seed if not args.baseline else None,
        "window": args.window if not args.baseline else None,
        "mode": args.mode if not args.baseline else None,
        "n_options": n_options,
        "n_questions": len(ds),
        "thinking_level": args.thinking_level,
        "limit": args.limit,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "timestamp": now_str,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nRun config saved: {output_dir}/run_config.json")

    retrieve_cmd = f"python3 src/retrieve_inference_vertex_batch.py --output-dir {output_dir}"
    with open(os.path.join(output_dir, "retrieve_command.txt"), "w") as f:
        f.write(retrieve_cmd + "\n")

    print(f"\nDone. Retrieve results with:")
    print(f"  {retrieve_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a Vertex AI Gemini batch inference job on one augmented MMLU-Pro configuration."
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
        help="Placement mode: 'block' or 'sparse'.",
    )
    parser.add_argument(
        "--vertex-model",
        type=str,
        default="gemini-3-flash-preview",
        help="Vertex AI Gemini model name (e.g. 'gemini-3-flash-preview', 'gemini-3.1-pro-preview').",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP project ID. Falls back to VERTEX_PROJECT env var.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="global",
        help="Vertex AI API location (default: global).",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="GCS bucket name (without gs://). Falls back to GCS_BUCKET env var.",
    )
    parser.add_argument(
        "--n-options",
        type=int,
        default=1000,
        help="Number of options in the augmented dataset (e.g. 100, 500, 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Distractor sampling seed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of questions (useful for smoke tests).",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run on the original 10-option baseline. Ignores --window, --mode, --seed, --n-options.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions",
        help="Base directory for output files.",
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="HIGH",
        choices=VALID_THINKING_LEVELS,
        help=(
            "Gemini 3 thinking level (default: HIGH). "
            "MINIMAL/LOW = faster, fewer tokens; MEDIUM = balanced; HIGH = deep reasoning. "
            "Gemini 3 Pro always thinks; level controls depth, not on/off."
        ),
    )

    args = parser.parse_args()

    if not args.bucket:
        args.bucket = os.getenv("GCS_BUCKET")
    if not args.bucket:
        print("ERROR: --bucket or GCS_BUCKET env var required.", file=sys.stderr)
        sys.exit(1)

    main(args)
