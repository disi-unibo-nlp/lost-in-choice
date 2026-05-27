"""
Analyze reasoning token usage across N-options, option dependency, placement mode,
window position, and correctness — per model and pooled across models.

For every completed run under --completions-dir the script reads:
    raw_completions.jsonl        — completion_tokens (usage field)
    raw_completions_gemini.jsonl — usageMetadata (fallback for Gemini runs)
    predictions.jsonl            — correct, option_dependent_vs_independent
    run_config.json              — model, n_options, window, mode, seed

completion_tokens includes reasoning + answer tokens. For Gemini this is
thoughtsTokenCount + candidatesTokenCount from usageMetadata.

When multiple timestamps exist for the same (model, dataset_tag), the latest is used.

Usage:
    python src/analysis/analyze_reasoning.py \\
        --completions-dir out/completions

    # restrict to specific models:
    python src/analysis/analyze_reasoning.py \\
        --completions-dir out/completions \\
        --models gpt-oss-120b gemini-3-flash-preview

Outputs (written to --out-dir, default out/reasoning_analysis/):
    reasoning_records.jsonl  — flat per-question records for downstream use
    reasoning_stats.json     — all aggregated tables
    reasoning_report.md      — human-readable Markdown summary
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median as _median
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATASET_TAG_RE = re.compile(
    r"mmlu-pro-augmented-(?P<n_options>\d+)__seed_(?P<seed>\d+)"
    r"__window_(?P<window>[\d.]+)__(?P<mode>block|sparse)"
)


def _stats(values: list) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None, "p95": None}
    arr = sorted(values)
    n = len(arr)
    return {
        "n": n,
        "mean": round(sum(arr) / n, 1),
        "median": round(_median(arr), 1),
        "p25": round(arr[max(0, int(n * 0.25) - 1)], 1),
        "p75": round(arr[min(n - 1, int(n * 0.75))], 1),
        "p95": round(arr[min(n - 1, int(n * 0.95))], 1),
    }


def _group_stats(records: list, group_key) -> dict:
    buckets: dict = defaultdict(list)
    for r in records:
        val = str(r[group_key]) if group_key in r else "unknown"
        buckets[val].append(r["completion_tokens"])

    try:
        sorted_keys = sorted(buckets.keys(), key=lambda x: float(x))
    except ValueError:
        sorted_keys = sorted(buckets.keys())

    return {k: _stats(buckets[k]) for k in sorted_keys}


def _correctness_stats(records: list) -> dict:
    buckets: dict = defaultdict(list)
    for r in records:
        label = {True: "correct", False: "incorrect", None: "unparsed"}[r["correct"]]
        buckets[label].append(r["completion_tokens"])
    return {
        k: _stats(buckets[k])
        for k in ("correct", "incorrect", "unparsed") if k in buckets
    }


def _dim_x_model(records: list, dim_key: str) -> dict:
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        dim_val = str(r.get(dim_key, "unknown"))
        buckets[dim_val][r["model_slug"]].append(r["completion_tokens"])

    try:
        sorted_keys = sorted(buckets.keys(), key=lambda x: float(x))
    except ValueError:
        sorted_keys = sorted(buckets.keys())

    return {
        dim_val: {model: _stats(vals) for model, vals in buckets[dim_val].items()}
        for dim_val in sorted_keys
    }


def compute_stats(records: list) -> dict:
    return {
        "n":                    len(records),
        "x_model":              _group_stats(records, "model_slug"),
        "x_correctness":        _correctness_stats(records),
        "x_n_options_x_model":  _dim_x_model(records, "n_options"),
        "x_mode_x_model":       _dim_x_model(records, "mode"),
        "x_window_x_model":     _dim_x_model(records, "window"),
        "x_option_dep_x_model": _dim_x_model(records, "option_dep"),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _latest_run_dir(dataset_dir: Path) -> Optional[Path]:
    ts_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    for d in reversed(ts_dirs):
        if (d / "raw_completions.jsonl").exists() and (d / "run_config.json").exists():
            return d
    return None


def load_records(completions_dir: Path, models_filter: Optional[list]) -> list:
    records = []
    skipped = 0

    for model_dir in sorted(completions_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name
        if models_filter and model_slug not in models_filter:
            continue

        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            m = DATASET_TAG_RE.match(dataset_dir.name)
            if not m:
                continue

            run_dir = _latest_run_dir(dataset_dir)
            if run_dir is None:
                skipped += 1
                continue

            cfg_path  = run_dir / "run_config.json"
            raw_path  = run_dir / "raw_completions.jsonl"
            pred_path = run_dir / "predictions.jsonl"

            with open(cfg_path) as f:
                cfg = json.load(f)

            # Build Gemini usage lookup (fallback when OpenAI-compat file lacks usage)
            gemini_usage: dict = {}
            gemini_path = run_dir / "raw_completions_gemini.jsonl"
            if gemini_path.exists():
                with open(gemini_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        cid = str(rec.get("custom_id", ""))
                        meta = rec.get("response", {}).get("usageMetadata", {})
                        gemini_usage[cid] = (
                            meta.get("thoughtsTokenCount", 0)
                            + meta.get("candidatesTokenCount", 0)
                        )

            # Build question-level prediction lookup
            pred_lookup: dict = {}
            if pred_path.exists():
                with open(pred_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        qid = str(rec["question_id"])
                        pred_lookup[qid] = {
                            "correct":    rec.get("correct"),
                            "option_dep": rec.get("option_dependent_vs_independent", "unknown"),
                        }

            n_options = int(m.group("n_options"))
            window    = float(m.group("window"))
            mode      = m.group("mode")

            bad_lines = 0
            with open(raw_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        bad_lines += 1
                        continue
                    qid  = str(item.get("custom_id", ""))
                    body = item.get("response", {}).get("body", {})

                    comp_tk = (body.get("usage", {}).get("completion_tokens")
                               or item.get("n_completion_tokens")
                               or 0)
                    if not comp_tk and qid in gemini_usage:
                        comp_tk = gemini_usage[qid]

                    pred = pred_lookup.get(qid, {})
                    records.append({
                        "question_id":       qid,
                        "model_slug":        model_slug,
                        "model_name":        cfg.get("model_name", model_slug),
                        "n_options":         n_options,
                        "window":            window,
                        "mode":              mode,
                        "seed":              cfg.get("seed"),
                        "completion_tokens": comp_tk,
                        "correct":           pred.get("correct"),
                        "option_dep":        pred.get("option_dep", "unknown"),
                        "run_dir":           str(run_dir),
                    })
            if bad_lines:
                print(f"  WARNING: {bad_lines} malformed JSON lines skipped in {raw_path}")

    print(f"Loaded {len(records)} question records  ({skipped} run dirs skipped — missing files)")
    return records


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _mean_or_dash(cell: Optional[dict]) -> str:
    if cell is None or cell.get("mean") is None:
        return "—"
    return f"{cell['mean']:.0f}"


def _pivot_table(title: str, dim_data: dict, dim_label: str) -> list:
    models = sorted({m for row in dim_data.values() for m in row.keys()})
    out = [
        f"## {title}\n",
        f"| {dim_label} | " + " | ".join(models) + " |",
        "|---|" + "---|" * len(models),
    ]
    for dim_val, model_map in dim_data.items():
        cells = " | ".join(_mean_or_dash(model_map.get(m)) for m in models)
        out.append(f"| {dim_val} | {cells} |")
    out.append("")
    return out


def render_report(stats: dict) -> str:
    lines = [
        "# Reasoning Token Analysis — Per Model\n",
        "_Auto-generated by analyze_reasoning.py_\n",
        f"Total question-run records: **{stats['n']}**\n",
        "_All cells show mean completion_tokens._\n",
    ]

    lines += _pivot_table("Comp-tokens Mean: N Options × Model",       stats["x_n_options_x_model"], "N options")
    lines += _pivot_table("Comp-tokens Mean: Placement Mode × Model",  stats["x_mode_x_model"],      "Mode")
    lines += _pivot_table("Comp-tokens Mean: Window Position × Model", stats["x_window_x_model"],    "Window")
    lines += _pivot_table("Comp-tokens Mean: Option Dependency × Model", stats["x_option_dep_x_model"], "Option dep")

    lines += ["## Overall Per Model\n",
              "| Model | N | mean | median | p95 |",
              "|---|---|---|---|---|"]
    for model, ct in stats["x_model"].items():
        lines.append(f"| {model} | {ct['n']} | {ct['mean'] or '—'} | {ct['median'] or '—'} | {ct['p95'] or '—'} |")
    lines.append("")

    lines += ["## Comp-tokens Mean: Correctness (pooled)\n",
              "| Outcome | N | mean |",
              "|---|---|---|"]
    for label, ct in stats["x_correctness"].items():
        lines.append(f"| {label} | {ct['n']} | {ct['mean'] or '—'} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    completions_dir = Path(args.completions_dir)
    if not completions_dir.is_dir():
        raise SystemExit(f"Not a directory: {completions_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_filter = [Path(m).name for m in args.models] if args.models else None

    records = load_records(completions_dir, models_filter)
    if not records:
        raise SystemExit("No records found — check --completions-dir and that runs have raw_completions.jsonl")

    records_path = out_dir / "reasoning_records.jsonl"
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Records saved → {records_path}")

    stats = compute_stats(records)

    stats_path = out_dir / "reasoning_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats   saved → {stats_path}")

    report_path = out_dir / "reasoning_report.md"
    with open(report_path, "w") as f:
        f.write(render_report(stats))
    print(f"Report  saved → {report_path}")

    print(f"\n{'='*60}")
    print(f"Total records: {stats['n']}")

    models = sorted(stats["x_model"].keys())
    print(f"\nMean completion_tokens × n_options × model:")
    for n_opt, model_map in stats["x_n_options_x_model"].items():
        row = "  ".join(f"{m}={model_map.get(m, {}).get('mean', '—')}" for m in models)
        print(f"  {n_opt:>6} options  {row}")

    print(f"\nMean completion_tokens × mode × model:")
    for mode, model_map in stats["x_mode_x_model"].items():
        row = "  ".join(f"{m}={model_map.get(m, {}).get('mean', '—')}" for m in models)
        print(f"  {mode:<8}  {row}")

    print(f"\nMean completion_tokens × option_dep × model:")
    for dep, model_map in stats["x_option_dep_x_model"].items():
        row = "  ".join(f"{m}={model_map.get(m, {}).get('mean', '—')}" for m in models)
        print(f"  {dep:<26}  {row}")

    print(f"\nMean completion_tokens × correctness:")
    for label, ct in stats["x_correctness"].items():
        print(f"  {label:<12}  mean={ct['mean']}  (n={ct['n']})")

    print(f"\nOverall per model:")
    for model, ct in stats["x_model"].items():
        print(f"  {model:<40}  mean={ct['mean']}  median={ct['median']}  (n={ct['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze reasoning token usage across experiment dimensions."
    )
    parser.add_argument(
        "--completions-dir",
        default="out/completions",
        help="Root completions directory (default: out/completions).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_SLUG",
        help="Restrict analysis to these model slugs (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        default="out/reasoning_analysis",
        help="Output directory (default: out/reasoning_analysis).",
    )
    args = parser.parse_args()
    main(args)
