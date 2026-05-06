"""
Pilot: distractor safety judge on the first 50 questions.

For each question takes the first 50 distractors, splits them into 5 batches
of 10, and calls the judge inline.  The judge identifies unsafe distractors
AND proposes a safe replacement in a single call.

Usage:
    python preprocess/analysis/pilot_distractor_judge.py
    python preprocess/analysis/pilot_distractor_judge.py \\
        --data data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl \\
        --n-questions 50 --batches-per-question 5
"""

import argparse
import json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from together import Together

load_dotenv()


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class FlaggedOption(BaseModel):
    id: int              # 0-based index within the batch
    reason: str          # why this option is unsafe
    replacement: str     # a safe replacement written by the model


class JudgeResponse(BaseModel):
    flagged: List[FlaggedOption]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating answer options for a multiple-choice question.

QUESTION:
{question}

CORRECT ANSWER (for reference only):
{gold}

CANDIDATE OPTIONS (labelled 0–99):
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
    {{"id": <0-based index within this batch>, "reason": "<one sentence>", "replacement": "<new safe option text>"}}
  ]
}}

Return {{"flagged": []}} if all options are safe.\
"""


def format_options_block(options: list) -> str:
    return "\n".join(f"({i}) {opt}" for i, opt in enumerate(options))


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_judge(
    client: Together,
    model: str,
    question: str,
    gold: str,
    batch: list,
) -> tuple[JudgeResponse, str]:
    """Return (parsed_response, raw_json_string)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold=gold,
        options_block=format_options_block(batch),
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5500,
        temperature=0,
        response_format={
            "type": "json_object",
            "schema": JudgeResponse.model_json_schema(),
        },
    )
    raw = response.choices[0].message.content.strip()
    return JudgeResponse.model_validate_json(raw), raw


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, limit: int) -> list:
    records = []
    with open(path) as f:
        for line in f:
            if len(records) >= limit:
                break
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    client   = Together()
    data     = load_jsonl(Path(args.data), args.n_questions)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_flagged = 0
    total_checked = 0

    with open(out_path, "w") as out_f:
        for q_idx, ex in enumerate(data):
            qid          = ex["question_id"]
            question     = ex["question"]
            orig_options = ex["options"]
            gold         = orig_options[ex["answer_index"]]
            orig_set     = set(orig_options)

            distractors = [o for o in ex["new_options"] if o not in orig_set]
            n_sample    = args.batches_per_question * 100
            sample      = distractors[:n_sample]

            print(f"\n{'='*60}")
            print(f"Q {q_idx+1}/{len(data)}  id={qid}  [{ex.get('category','')}]  "
                  f"[{ex.get('option_dependent_vs_independent','')}]")
            print(f"  {question[:100]}")
            print(f"  GOLD: {gold!r}")
            print(f"  Checking {len(sample)} distractors in {args.batches_per_question} batches of 100")

            q_record = {
                "question_id": qid,
                "question"   : question,
                "gold"       : gold,
                "category"   : ex.get("category", ""),
                "opt_dep"    : ex.get("option_dependent_vs_independent", ""),
                "batches"    : [],
            }

            q_flagged = 0
            for b_idx in range(args.batches_per_question):
                batch = sample[b_idx*100 : (b_idx+1)*100]
                if not batch:
                    break
                result, raw = call_judge(client, args.model, question, gold, batch)
                total_checked += len(batch)

                batch_record = {
                    "batch_idx"     : b_idx,
                    "n_options"     : len(batch),
                    "raw_response"  : raw,
                    "flagged"       : [item.model_dump() for item in result.flagged],
                }
                q_record["batches"].append(batch_record)

                if not result.flagged:
                    print(f"  [batch {b_idx}] all {len(batch)} safe ✓")
                else:
                    for item in result.flagged:
                        global_idx = b_idx * 100 + item.id
                        opt_text   = batch[item.id] if item.id < len(batch) else "???"
                        print(f"  [batch {b_idx}] ⚠ UNSAFE  batch_idx={item.id}  global_idx={global_idx}")
                        print(f"    original    : {opt_text!r}")
                        print(f"    reason      : {item.reason}")
                        print(f"    replacement : {item.replacement!r}")
                        q_flagged  += 1
                        total_flagged += 1

            print(f"  → {q_flagged} flagged in this question")
            out_f.write(json.dumps(q_record) + "\n")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Questions checked  : {len(data)}")
    print(f"  Distractors checked: {total_checked}")
    print(f"  Unsafe flagged     : {total_flagged}  "
          f"({100*total_flagged/total_checked:.1f}% of checked)")
    print(f"  Results saved to   : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=
        "data/bench/mmlu-pro-augmented-1000/seed_42/window_0.75/block.jsonl")
    parser.add_argument("--model", default="OpenAI/gpt-oss-20B")
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--batches-per-question", type=int, default=10,
                        help="Batches of 100 distractors per question (10 covers all 990)")
    parser.add_argument("--output", default="out/distractor_judge/pilot_results.jsonl",
                        help="Path to save raw + parsed results as JSONL")
    args = parser.parse_args()
    main(args)
