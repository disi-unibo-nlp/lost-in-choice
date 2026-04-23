from datasets import load_from_disk
import json
import os
import argparse
from dotenv import load_dotenv
from datetime import datetime
from together import Together

load_dotenv()

# --- PROMPT TEMPLATE ---

DEPENDENCY_PROMPT_TEMPLATE = """Decide whether a knowledgeable person could answer the question correctly
WITHOUT seeing the answer options.

Core test — mentally remove the options and ask the question as open-ended:
  • If it is answerable (a specific fact, definition, formula, calculation, etc.) → option_independent
  • If the answer is ambiguous or requires comparing the options             → option_dependent

Additional rules:
  1. Phrasing like "which of the following" or "which action" does NOT automatically
     make a question option-dependent. Ask whether the open-ended version is still
     answerable.
  2. Questions that ask for a best/most-likely/least answer among plausible alternatives,
     or that use "EXCEPT / NOT / LEAST", are almost always option-dependent because
     the comparison is the task.
  3. When the options introduce an evaluation category that is NOT implied by the question
     itself — such as "Not Given", "Cannot be determined", "All of the above", or composite
     True/False/Not-Given judgements — the question is option-dependent. The options are
     redefining what a correct answer even looks like, so they cannot be ignored.
  4. When in doubt, apply the core test above and prefer the label that reflects
     whether domain knowledge alone is sufficient.

---

Examples:

Question: What is the powerhouse of the cell?
Options: (0) Nucleus  (1) Mitochondria  (2) Ribosome  (3) Golgi apparatus
Reasoning: A knowledgeable person answers "mitochondria" directly — no options needed.
{"label": "option_independent"}

---

Question: Which of the following best explains why ice floats on water?
Options: (0) Ice is colder than water  (1) Ice has a lower density than water  (2) Ice expands when it freezes  (3) Water contracts when cooled
Reasoning: The direct answer is "ice has lower density than water" — answerable without options despite the "which of the following" phrasing.
{"label": "option_independent"}

---

Question: Which of the following medications is least likely to cause drowsiness?
Options: (0) Diphenhydramine  (1) Loratadine  (2) Promethazine  (3) Hydroxyzine
Reasoning: "Least likely" requires comparing the side-effect profiles of all four options against each other — the options define the task.
{"label": "option_dependent"}

---

Question: Which of the following is NOT a function of the liver?
Options: (0) Bile production  (1) Insulin secretion  (2) Detoxification  (3) Glycogen storage
Reasoning: The NOT framing requires scanning all options to identify the outlier — cannot be answered without them.
{"label": "option_dependent"}

---

Question: Statement 1| VGGNets have convolutional kernels of smaller width and height than AlexNet's first-layer kernels. Statement 2| Data-dependent weight initialization procedures were introduced before Batch Normalization.
Options: (0) Both Not Given  (1) Not Given, True  (2) True, True  (3) False, Not Given  (4) True, False  (5) False, False  (6) True, Not Given  (7) False, True  (8) Not Given, False  (9) Not Given, Not Given
Reasoning: Without the options, a knowledgeable person would only consider True/False for each statement. The options introduce "Not Given" as a third evaluation category that is invisible from the question alone — the options redefine what a correct answer looks like.
{"label": "option_dependent"}

---

Now annotate the following:

Question:
<QUESTION>

Options:
<OPTIONS>

Return ONLY valid JSON, no explanation, no markdown, **strictly adhering to the format below**:

{"label": "option_dependent" or "option_independent"}
"""

# DEPENDENCY_PROMPT_TEMPLATE = """You are annotating exam-style multiple-choice questions.

# Your task: decide whether a knowledgeable person could answer the question correctly
# WITHOUT seeing the answer options.

# Core test — mentally remove the options and ask the question as open-ended:
#   • If it is answerable (a specific fact, definition, formula, calculation, etc.) → option_independent
#   • If the answer is ambiguous or requires comparing the options             → option_dependent

# Additional rules:
#   1. Phrasing like "which of the following" or "which action" does NOT automatically
#      make a question option-dependent. Ask whether the open-ended version is still
#      answerable.
#   2. Questions that ask for a best/most-likely/least answer among plausible alternatives,
#      or that use "EXCEPT / NOT / LEAST", are almost always option-dependent because
#      the comparison is the task.
#   3. When the options introduce an evaluation category that is NOT implied by the question
#      itself — such as "Not Given", "Cannot be determined", "All of the above", or composite
#      True/False/Not-Given judgements — the question is option-dependent. The options are
#      redefining what a correct answer even looks like, so they cannot be ignored.
#   4. When in doubt, apply the core test above and prefer the label that reflects
#      whether domain knowledge alone is sufficient.

# ---

# Now annotate the following:

# Question:
# <QUESTION>

# Options:
# <OPTIONS>

# Return ONLY valid JSON, no explanation, no markdown:

# {"label": "option_dependent" or "option_independent"}"""



# --- HELPER FUNCTIONS ---

def format_prompt(item):
    question = item['question']
    options = item['options']
    item_id = str(item.get('question_id', hash(question)))

    formatted_options = []
    for i, option in enumerate(options):
        formatted_options.append(f"({i}) {option}")
    options_string = "\n".join(formatted_options)

    prompt_dependency = DEPENDENCY_PROMPT_TEMPLATE.replace("<QUESTION>", question).replace("<OPTIONS>", options_string)

    return (item_id, prompt_dependency)

def create_batch_together(benchmark, output_dir, model_name_path, client):
    model_name = model_name_path.split("/")[-1]
    jsonl_path = f"{output_dir}/batch_together_{model_name}.jsonl"

    with open(jsonl_path, "w") as f:
        for i, item in enumerate(benchmark):
            item_id, prompt_dependency = format_prompt(item)

            # print the first few prompts for sanity check
            if i < 3:
                print(f"\n--- Sample {i+1} ---")
                print("Question:", item['question'])
                print("Options:", item['options'])
                print("Formatted Prompt:\n", prompt_dependency)
                print("\n-------------------\n")

            req_dependency = {
                "custom_id": f"{item_id}_dependency",
                "body": {
                    "model": model_name_path,
                    "messages": [{"role": "user", "content": prompt_dependency}],
                    #"temperature": 0,
                    "max_tokens": 16000,
                    "response_format": {
                        "type": "regex",
                        "pattern": "(option_dependent|option_independent)"
                    },
                }
            }

            f.write(json.dumps(req_dependency) + "\n")

    print(f"Uploading file: {jsonl_path}")
    file_resp = client.files.upload(file=jsonl_path, purpose="batch-api", check=False)
    print("File upload response:", file_resp.id)

    batch = client.batches.create(
        input_file_id=file_resp.id, endpoint="/v1/chat/completions"
    )

    batch_stat = client.batches.retrieve(batch.job.id)

    print(f"\n--- BATCH INITIATED ---")
    print(f"Job ID: {batch.job.id}")
    with open(f"{output_dir}/job_id.txt", "w") as f:
        f.write(f"{batch.job.id}")

    print(f"Current Status: {batch_stat.status}")
    print(f"Results will be saved in: {output_dir}")


# --- MAIN BLOCK ---

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run batch dependency categorization on MMLU dataset.")

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="mmlu_pro_downsampled_2858",
        help="Local directory path to the saved downsampled dataset."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen/Qwen3.5-9B",
        help="Together AI model path to use for categorization."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/completions",
        help="Base output directory to save requests and job IDs."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process (for quick testing)."
    )

    args = parser.parse_args()

    now = datetime.now()
    now_dir = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{args.output_dir}/{args.model_name.split('/')[-1]}/categorization/{now_dir}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from './{args.dataset_dir}'...")
    benchmark = load_from_disk(args.dataset_dir)

    if args.limit:
        benchmark = benchmark.select(range(args.limit))

    print(f"Total samples to process: {len(benchmark)}")
    print(f"Total API requests generated: {len(benchmark)}")

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    if not TOGETHER_API_KEY:
        raise ValueError("TOGETHER_API_KEY not found in environment variables. Set it using `export TOGETHER_API_KEY=your_key`")

    client = Together(api_key=TOGETHER_API_KEY)

    create_batch_together(
        benchmark=benchmark,
        output_dir=output_dir,
        model_name_path=args.model_name,
        client=client
    )
