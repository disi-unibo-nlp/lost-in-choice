"""
Retrieve and parse results from a completed strategy-classification batch job.

Joins API responses with manifest.json and writes classified_traces.jsonl —
one record per trace with the assigned taxonomy code, confidence, and rationale.

Usage:
    python preprocess/analysis/retrieve_classification_batch.py \\
        --output-dir out/classification_batches/2026-04-30_12-00-00

    python preprocess/analysis/retrieve_classification_batch.py \\
        --output-dir out/classification_batches/2026-04-30_12-00-00 \\
        --job-id <BATCH_JOB_ID>
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from together import Together

sys.path.insert(0, str(Path(__file__).parent))
from pilot_taxonomy_judge import Classification

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
        raise SystemExit(f"manifest.json not found in {out_dir}.")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"Manifest: {len(manifest)} requests")

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    batch = client.batches.retrieve(job_id)
    print(f"Status: {batch.status}")
    if batch.status != "COMPLETED":
        print("Job not completed yet — re-run when done.")
        return

    # Download raw completions
    raw_path = out_dir / "raw_completions.jsonl"
    if raw_path.exists():
        print(f"Using existing {raw_path}")
    else:
        print(f"Downloading → {raw_path}")
        with client.files.with_streaming_response.content(id=batch.output_file_id) as resp:
            with open(raw_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        print("Download complete.")

    # Parse and join with manifest
    out_path = out_dir / "classified_traces.jsonl"
    n_ok = n_fail = 0
    confidence_counts: Counter = Counter()
    strategy_counts: Counter = Counter()

    with open(raw_path) as fin, open(out_path, "w") as fout:
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
                clf = Classification.model_validate_json(content)
                record = {
                    **meta,
                    "primary_strategy": clf.primary_strategy,
                    "secondary_strategy": clf.secondary_strategy,
                    "confidence": clf.confidence,
                    "rationale": clf.rationale,
                    "raw_response": content,
                }
                fout.write(json.dumps(record) + "\n")
                n_ok += 1
                confidence_counts[clf.confidence] += 1
                strategy_counts[clf.primary_strategy] += 1
            except Exception as exc:
                print(f"  PARSE ERROR [{custom_id}]: {exc}")
                n_fail += 1

    print(f"\nParsed: {n_ok} OK, {n_fail} failed → {out_path}")
    print(f"\nConfidence breakdown:")
    for lvl in ("high", "medium", "low"):
        n = confidence_counts[lvl]
        print(f"  {lvl:<8} {n:4d}  ({100*n/max(n_ok,1):.1f}%)")

    print(f"\nStrategy distribution (primary, all confidence levels):")
    for code, count in strategy_counts.most_common():
        print(f"  {code:<6} {count:4d}  ({100*count/max(n_ok,1):.1f}%)")

    if n_ok > 0:
        analyze_cmd = (
            f"python preprocess/analysis/analyze_strategies.py "
            f"--classified-path {out_path}"
        )
        print(f"\nRun analysis with:\n  {analyze_cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve classification batch results and write classified_traces.jsonl."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory created by submit_classification_batch.py.",
    )
    parser.add_argument(
        "--job-id", type=str, default=None,
        help="Together AI batch job ID (optional if job_id.txt exists).",
    )
    args = parser.parse_args()
    main(args)
