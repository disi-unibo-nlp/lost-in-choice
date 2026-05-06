"""
Retrieve and evaluate a completed Vertex AI Gemini batch inference job.

Usage:
    python src/retrieve_inference_vertex_batch.py \\
        --output-dir out/completions/gemini-2-5-flash/mmlu-pro-augmented-1000__seed_42__window_0.50__sparse/2026-05-04_10-00-00

    # Job name is read from job_name.txt in --output-dir automatically.
    # Project is read from run_config.json or VERTEX_PROJECT env var.

Outputs (written into --output-dir):
    raw_completions_gemini.jsonl  — raw lines downloaded from GCS (native Gemini format)
    raw_completions.jsonl         — normalized to Together-compatible format
    predictions.jsonl             — identical format to retrieve_inference.py
    metrics.json                  — identical format to retrieve_inference.py
"""

import argparse
import json
import os
import sys

import fsspec
from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

load_dotenv()

# Reuse evaluate() from retrieve_inference.py to guarantee identical output format.
sys.path.insert(0, os.path.dirname(__file__))
from retrieve_inference import evaluate


# ---------------------------------------------------------------------------
# Download from GCS
# ---------------------------------------------------------------------------

def download_gemini_output(
    job_name: str,
    run_config: dict,
    output_dir: str,
    project: str,
    location: str,
) -> str:
    """Poll job state, download predictions.jsonl from GCS. Returns local path."""
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=HttpOptions(api_version="v1"),
    )

    print(f"Checking batch job: {job_name}")
    job = client.batches.get(name=job_name)
    print(f"  State: {job.state}")

    if job.state in ("JOB_STATE_RUNNING", "JOB_STATE_PENDING", "JOB_STATE_QUEUED"):
        print("Job is still running. Re-run this script once it completes.", file=sys.stderr)
        sys.exit(1)

    if job.state != "JOB_STATE_SUCCEEDED":
        err = getattr(job, "error", "unknown error")
        print(f"Job did not succeed (state={job.state}): {err}", file=sys.stderr)
        sys.exit(1)

    # Resolve output GCS URI
    gcs_output_uri = run_config.get("gcs_output_uri")
    if not gcs_output_uri and hasattr(job, "dest") and job.dest:
        gcs_output_uri = job.dest.gcs_uri
    if not gcs_output_uri:
        print("ERROR: cannot determine GCS output URI.", file=sys.stderr)
        sys.exit(1)

    fs = fsspec.filesystem("gcs")

    # Vertex AI writes output to a subdirectory: {dest}/*/predictions.jsonl
    bare = gcs_output_uri.removeprefix("gs://")
    file_paths = fs.glob(f"{bare}/*/predictions.jsonl")
    if not file_paths:
        file_paths = fs.glob(f"{bare}/predictions.jsonl")
    if not file_paths:
        print(f"ERROR: no predictions.jsonl found under {gcs_output_uri}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading from: gs://{file_paths[0]}")
    raw_gemini_path = os.path.join(output_dir, "raw_completions_gemini.jsonl")
    with fs.open(file_paths[0], "r") as src, open(raw_gemini_path, "w") as dst:
        for line in src:
            dst.write(line if line.endswith("\n") else line + "\n")

    print(f"Saved raw Gemini output: {raw_gemini_path}")
    return raw_gemini_path


# ---------------------------------------------------------------------------
# Normalize Gemini → Together-compatible raw_completions.jsonl
# ---------------------------------------------------------------------------

def normalize_gemini_completions(
    raw_gemini_path: str,
    output_dir: str,
    qid_order: list,
) -> str:
    """
    Convert native Gemini batch output to the Together-compatible format that
    retrieve_inference.py's evaluate() expects:

        {"custom_id": "...", "response": {"body": {"choices": [{"message": {"content": "...", "reasoning": "..."}}]}}}

    Gemini thinking parts (part["thought"] == True) are mapped to "reasoning".
    The JSON answer text goes into "content".

    Falls back to qid_order (by position) if Gemini does not echo custom_id.
    """
    with open(raw_gemini_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]

    normalized = []
    for i, row in enumerate(lines):
        custom_id = row.get("custom_id")
        if not custom_id:
            custom_id = qid_order[i] if i < len(qid_order) else f"unknown_{i}"

        candidates = row.get("response", {}).get("candidates", [])
        content = ""
        reasoning = ""

        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                text = part.get("text", "")
                if part.get("thought"):
                    reasoning += text
                else:
                    content += text

        normalized.append({
            "custom_id": custom_id,
            "response": {
                "body": {
                    "choices": [{
                        "message": {
                            "content": content,
                            "reasoning": reasoning or None,
                        }
                    }]
                }
            },
        })

    raw_path = os.path.join(output_dir, "raw_completions.jsonl")
    with open(raw_path, "w") as f:
        for rec in normalized:
            f.write(json.dumps(rec) + "\n")

    print(f"Normalized {len(normalized)} completions → {raw_path}")
    return raw_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"ERROR: output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Load run config
    config_path = os.path.join(output_dir, "run_config.json")
    run_config: dict = {}
    if os.path.isfile(config_path):
        with open(config_path) as f:
            run_config = json.load(f)
        print(f"Loaded run config: {config_path}")
    else:
        print("WARNING: run_config.json not found — metadata will be limited.")

    # Resolve job name
    job_name = args.job_name
    if not job_name:
        name_file = os.path.join(output_dir, "job_name.txt")
        if os.path.isfile(name_file):
            with open(name_file) as f:
                job_name = f.read().strip()
            print(f"Loaded job name: {job_name}")
        else:
            print("ERROR: --job-name not provided and job_name.txt not found.", file=sys.stderr)
            sys.exit(1)

    # Load question order for positional fallback
    qid_order: list = []
    order_file = os.path.join(output_dir, "question_order.json")
    if os.path.isfile(order_file):
        with open(order_file) as f:
            qid_order = json.load(f)

    # Resolve project / location
    project = args.project or run_config.get("vertex_project") or os.getenv("VERTEX_PROJECT")
    location = args.location or run_config.get("vertex_location", "global")
    if not project:
        print("ERROR: GCP project required. Use --project or VERTEX_PROJECT env var.", file=sys.stderr)
        sys.exit(1)

    # Download from GCS
    raw_gemini_path = download_gemini_output(job_name, run_config, output_dir, project, location)

    # Normalize to Together-compatible format
    raw_path = normalize_gemini_completions(raw_gemini_path, output_dir, qid_order)

    # Evaluate — identical logic and output format as retrieve_inference.py
    evaluate(raw_path, output_dir, run_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve and evaluate a completed Vertex AI Gemini batch inference job."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory written by run_inference_vertex_batch.py (contains run_config.json, job_name.txt).",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Vertex AI batch job resource name. If omitted, read from job_name.txt in --output-dir.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="GCP project ID. Falls back to run_config.json then VERTEX_PROJECT env var.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Vertex AI location. Falls back to run_config.json then 'global'.",
    )
    args = parser.parse_args()
    main(args)
