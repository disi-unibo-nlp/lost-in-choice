"""
Retrieve and parse results from a completed open-coding batch job.

Joins API responses with manifest.json (saved by submit_open_coding_batch.py)
to produce open_codes.jsonl in the same format expected by synthesize_taxonomy.py.

Usage:
    # job_id.txt is picked up automatically from --output-dir:
    python preprocess/analysis/retrieve_open_coding_batch.py \\
        --output-dir out/open_coding_batches/2026-04-30_10-00-00

    # or pass --job-id explicitly:
    python preprocess/analysis/retrieve_open_coding_batch.py \\
        --output-dir out/open_coding_batches/2026-04-30_10-00-00 \\
        --job-id <BATCH_JOB_ID>
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from together import Together

sys.path.insert(0, str(Path(__file__).parent))
from pilot_taxonomy_judge import OpenCode

load_dotenv()


def main(args):
    out_dir = Path(args.output_dir)
    if not out_dir.exists():
        raise SystemExit(f"Output directory not found: {out_dir}")

    # Resolve job ID
    job_id = args.job_id
    if not job_id:
        job_id_path = out_dir / "job_id.txt"
        if not job_id_path.exists():
            raise SystemExit(f"No --job-id given and {job_id_path} not found.")
        job_id = job_id_path.read_text().strip()
    print(f"Job ID: {job_id}")

    # Load manifest
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found in {out_dir} — was this dir created by submit_open_coding_batch.py?")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} requests")

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    # Check status
    batch = client.batches.retrieve(job_id)
    print(f"Status: {batch.status}")
    if batch.status != "COMPLETED":
        print("Job not completed yet — re-run this script once the job finishes.")
        return

    # Download raw completions (skip if already present)
    raw_path = out_dir / "raw_completions.jsonl"
    if raw_path.exists():
        print(f"Using existing {raw_path}")
    else:
        print(f"Downloading raw completions → {raw_path}")
        with client.files.with_streaming_response.content(id=batch.output_file_id) as resp:
            with open(raw_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        print("Download complete.")

    # Parse and join with manifest
    open_codes_path = out_dir / "open_codes.jsonl"
    n_ok = n_fail = 0

    with open(raw_path) as fin, open(open_codes_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")
            meta = manifest.get(custom_id)

            if meta is None:
                print(f"  WARNING: custom_id not in manifest: {custom_id}")
                n_fail += 1
                continue

            try:
                content = item["response"]["body"]["choices"][0]["message"]["content"]
                code = OpenCode.model_validate_json(content)
                record = {
                    **meta,
                    "strategy_label": code.strategy_label,
                    "strategy_description": code.strategy_description,
                    "confidence": code.confidence,
                    "raw_response": content,
                }
                fout.write(json.dumps(record) + "\n")
                n_ok += 1
            except Exception as exc:
                print(f"  PARSE ERROR [{custom_id}]: {exc}")
                n_fail += 1

    print(f"\nParsed: {n_ok} OK, {n_fail} failed → {open_codes_path}")

    if n_ok > 0:
        synth_cmd = (
            f"python preprocess/analysis/synthesize_taxonomy.py "
            f"--open-codes-path {open_codes_path}"
        )
        print(f"\nRun synthesis with:\n  {synth_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve open-coding batch results and write open_codes.jsonl."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory created by submit_open_coding_batch.py.",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Together AI batch job ID (optional if job_id.txt exists in --output-dir).",
    )
    args = parser.parse_args()
    main(args)
