"""
Synthesize a reasoning-strategy taxonomy from open_codes.jsonl using Gemini.

Reads open_codes.jsonl (produced by retrieve_open_coding_batch.py or
pilot_taxonomy_judge.py), builds the synthesis prompt with the real label
frequency table, and makes a single Gemini API call to produce the taxonomy.

Requires GOOGLE_API_KEY (or GEMINI_API_KEY) in .env.

Cost: one call, ~100K input tokens → ~$0.01 with gemini-2.5-flash-preview-05-20.

Usage:
    python preprocess/analysis/synthesize_taxonomy.py \\
        --open-codes-path out/open_coding_batches/2026-04-30_10-00-00/open_codes.jsonl

    python preprocess/analysis/synthesize_taxonomy.py \\
        --open-codes-path out/open_coding_batches/2026-04-30_10-00-00/open_codes.jsonl \\
        --gemini-model gemini-2.5-flash-preview-05-20
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from pilot_taxonomy_judge import (
    TaxonomySynthesis,
    build_synthesis_prompt,
    taxonomy_to_markdown,
)

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Gemini does not support $defs in responseSchema, so define inline.
TAXONOMY_SYNTHESIS_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "taxonomy": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "code":                 {"type": "STRING"},
                    "name":                 {"type": "STRING"},
                    "description":          {"type": "STRING"},
                    "behavioral_signature": {"type": "STRING"},
                    "frequency":            {"type": "STRING"},
                },
                "required": ["code", "name", "description", "behavioral_signature", "frequency"],
            },
        },
        "notes": {"type": "STRING"},
    },
    "required": ["taxonomy", "notes"],
}


def main(args):
    open_codes_path = Path(args.open_codes_path)
    if not open_codes_path.exists():
        raise SystemExit(f"File not found: {open_codes_path}")

    with open(open_codes_path) as f:
        all_codes = [json.loads(line) for line in f if line.strip()]

    if not all_codes:
        raise SystemExit("open_codes.jsonl is empty.")

    print(f"Loaded {len(all_codes)} open codes")

    n_options_range = "+".join(
        str(n) for n in sorted(set(c["n_options"] for c in all_codes))
    )
    models_desc = ", ".join(sorted(set(c["model_slug"] for c in all_codes)))
    print(f"Models: {models_desc}  |  N-options: {n_options_range}")

    prompt = build_synthesis_prompt(all_codes, n_options_range, models_desc)
    n_tokens_est = len(prompt) // 4
    print(f"Prompt: ~{len(prompt):,} chars (~{n_tokens_est:,} tokens estimated)")

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY (or GEMINI_API_KEY) not set in environment.")

    client = genai.Client(api_key=api_key)

    print(f"Calling {args.gemini_model}...")
    resp = client.models.generate_content(
        model=args.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TAXONOMY_SYNTHESIS_SCHEMA,
        ),
    )

    raw = resp.text
    tax = TaxonomySynthesis.model_validate_json(raw)

    out_dir = open_codes_path.parent
    tax_json_path = out_dir / "taxonomy_draft.json"
    tax_md_path   = out_dir / "taxonomy_draft.md"

    with open(tax_json_path, "w") as f:
        json.dump(tax.model_dump(), f, indent=2)

    md = taxonomy_to_markdown(tax)
    with open(tax_md_path, "w") as f:
        f.write(md)

    print(f"\nTaxonomy ({len(tax.taxonomy)} strategies) → {tax_json_path}")
    print(f"Markdown → {tax_md_path}")
    print("\n" + "=" * 60)
    print(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize reasoning-strategy taxonomy from open_codes.jsonl using Gemini."
    )
    parser.add_argument(
        "--open-codes-path",
        required=True,
        help="Path to open_codes.jsonl produced by retrieve_open_coding_batch.py.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_GEMINI_MODEL}).",
    )
    args = parser.parse_args()
    main(args)
