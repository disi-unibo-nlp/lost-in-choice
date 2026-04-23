import json
import argparse
import os
from datasets import load_from_disk
from together import Together
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

import re

_VALID_LABELS = {"option_dependent", "option_independent"}

def parse_label(content: str, reasoning: str = "") -> str:
    """Extract label value from model output.

    Handles:
    - Plain strings: "option_independent"
    - Raw JSON: {"label": "option_independent"}
    - Markdown-fenced JSON: ```json\n{"label": "option_independent"}\n```
    - Empty content (finish_reason=length): falls back to last label mention in reasoning
    """
    text = content.strip()
    if text:
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        for candidate in (fenced, text):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "label" in obj:
                    val = str(obj["label"]).strip().lower()
                    if val in _VALID_LABELS:
                        return val
            except (json.JSONDecodeError, ValueError):
                pass
        plain = fenced.lower() if fenced else text.lower()
        if plain in _VALID_LABELS:
            return plain

    # # Fallback: scan reasoning for last label mention (handles finish_reason=length)
    # if reasoning:
    #     matches = list(re.finditer(r'\b(option_dependent|option_independent)\b', reasoning, re.IGNORECASE))
    #     if matches:
    #         return matches[-1].group(1).lower()

    return ""


def retrieve_and_augment(job_name, output_dir, dataset_dir, client):
    # 1. Load original downsampled dataset
    print(f"Loading original dataset from: {dataset_dir}")
    benchmark = load_from_disk(dataset_dir)

    # 2. Check batch status and retrieve results
    batch = client.batches.retrieve(job_name)
    print(f"Batch Status: {batch.status}")

    if batch.status != "COMPLETED":
        print("Batch is not yet completed. Please try again later.")
        return

    raw_output_file = os.path.join(output_dir, "raw_completions.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading batch results...")
    with client.files.with_streaming_response.content(id=batch.output_file_id) as response:
        with open(raw_output_file, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    print(f"Raw completions saved to: {raw_output_file}")

    # 3. Parse completions into a mapping dictionary
    print("Parsing completions and extracting labels...")
    with open(raw_output_file, "r") as f:
        completions = [json.loads(line) for line in f.readlines()]

    annotations_map = {}

    for item in completions:
        custom_id = item['custom_id']
        id_item, task_type = custom_id.rsplit("_", 1)

        msg = item['response']['body']['choices'][0]['message']
        label = parse_label(msg.get('content', ''), msg.get('reasoning', ''))

        if not label:
            label = "missing"

        if id_item not in annotations_map:
            annotations_map[id_item] = {}

        if task_type == "dependency":
            annotations_map[id_item]['option_dependent_vs_independent'] = label

    # 4. Debug: print first 5 completions with original question + options
    print("\n--- Debug: First 5 Completions ---")
    id_to_item = {str(item.get('question_id', hash(item['question']))): item for item in benchmark}
    for i, (item_id, anns) in enumerate(list(annotations_map.items())[:10]):
        orig = id_to_item.get(item_id, {})
        print(f"\n[{i+1}] ID: {item_id}")
        print(f"  Question: {orig.get('question', 'N/A')}")
        options = orig.get('options', [])
        for j, opt in enumerate(options):
            print(f"    ({j}) {opt}")
        print(f"  Label: {anns.get('option_dependent_vs_independent', 'missing')}")
    print("\n-----------------------------------\n")

    # 5. Augment the dataset with the dependency label
    print("Augmenting the dataset with dependency annotations...")

    def add_annotations(example):
        item_id = str(example.get('question_id', hash(example['question'])))
        anns = annotations_map.get(item_id, {})

        d_val = anns.get('option_dependent_vs_independent', "missing")
        example['option_dependent_vs_independent'] = d_val if d_val else "missing"

        return example

    augmented_ds = benchmark.map(add_annotations)

    # 6. Save the final augmented dataset
    model_name = os.path.normpath(output_dir).split(os.sep)[2]
    augmented_dataset_path = os.path.join(f"{dataset_dir}_categorized", model_name)

    augmented_ds.save_to_disk(augmented_dataset_path)
    print(f"\nDone! Augmented dataset saved (HF format) to: ./{augmented_dataset_path}")

    jsonl_path = f"{augmented_dataset_path}.jsonl"
    augmented_ds.to_json(jsonl_path, orient="records", lines=True)
    print(f"Done! Augmented dataset saved (JSONL format) to: ./{jsonl_path}")

    # 7. Print dependency statistics
    df = augmented_ds.to_pandas()

    print("\n--- Option Dependency Statistics ---")
    print(df['option_dependent_vs_independent'].value_counts().to_string())

    total = len(df)
    for label, count in df['option_dependent_vs_independent'].value_counts().items():
        print(f"  {label}: {count} ({100 * count / total:.1f}%)")

    # 8. Statistics per category
    if 'category' in df.columns:
        print("\n--- Option Dependency Breakdown per Category ---")
        cat_dependency = pd.crosstab(df['category'], df['option_dependent_vs_independent'])
        print(cat_dependency.to_string())

        print("\n--- Option Dependency Rate per Category ---")
        cat_pct = cat_dependency.div(cat_dependency.sum(axis=1), axis=0) * 100
        print(cat_pct.round(1).to_string())
    else:
        print("Column 'category' not found in the dataset. Skipping category-level statistics.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve Together API batch results and augment dataset with dependency labels.")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="mmlu_pro_downsampled_2858",
        help="Local directory path to the original downsampled dataset."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions/gemma-4-31B-it/categorization/2026-04-14_16-31-33",
        help="Directory where raw_completions.jsonl will be saved (same folder as the batch input)."
    )

    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="The Together AI batch job ID to retrieve."
    )

    args = parser.parse_args()

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in environment variables.")

    client = Together(api_key=TOGETHER_API_KEY)

    retrieve_and_augment(
        job_name=args.job_id,
        output_dir=args.output_dir,
        dataset_dir=args.dataset_dir,
        client=client
    )


