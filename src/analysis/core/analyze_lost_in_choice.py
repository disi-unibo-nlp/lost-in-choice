"""
RQ2 — Lost in Choice: What Does the Model Pick When It Fails?

Primary source: ALL predictions.jsonl files under --completions-dir (one per model × condition).
Each record already contains: correct, hit_original_option, window, placement_mode, category,
option_dependent_vs_independent.

Enrichment:
  1. N=10 baseline predictions  → lost_in_choice flag (correct@10 but wrong@N).
  2. classified_traces.jsonl    → primary_strategy (joined where available).

option_type is derived from (correct, hit_original_option):
  gold          — correct == True
  wrong_original — correct == False, hit_original_option == True
  distractor    — correct == False, hit_original_option == False

Produces:
    lic_records.jsonl  — enriched per-question records for downstream use
    lic_stats.json     — all aggregated tables
    lic_report.md      — human-readable markdown summary

Usage:
    python src/analysis/analyze_lost_in_choice.py \\
        --completions-dir out/completions \\
        --classified-path out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_TAG_RE = re.compile(
    r"mmlu-pro-augmented-(?P<n>\d+)__seed_\d+__window_(?P<window>[\d.]+)__(?P<mode>block|sparse)"
)
CONDITION_RE = re.compile(
    r"^(?P<model>.+?)__n(?P<n>\d+)_w(?P<window>[\d.]+)_(?P<mode>block|sparse)$"
)
BASELINE_TAG = "mmlu-pro-augmented-10__baseline"

STRATEGY_ORDER = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]
OPTION_TYPES   = ["gold", "wrong_original", "distractor"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _latest_pred_file(dataset_dir: Path):
    ts_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
    for d in reversed(ts_dirs):
        p = d / "predictions.jsonl"
        if p.exists():
            return p
    return None


def load_all_predictions(completions_dir: Path, models_filter) -> list:
    """
    Walk every non-baseline predictions.jsonl and return a flat list of records.
    Each record has: model_slug, n_options, window, window_str, mode, question_id,
    correct, hit_original_option, option_dep, category, parse_failed.
    """
    records = []
    skipped = 0

    for model_dir in sorted(completions_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name
        if models_filter and slug not in models_filter:
            continue

        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            m = DATASET_TAG_RE.match(dataset_dir.name)
            if not m:
                continue

            n_opt      = int(m.group("n"))
            window_str = f"{float(m.group('window')):.2f}"
            mode       = m.group("mode")

            pred_file = _latest_pred_file(dataset_dir)
            if pred_file is None:
                skipped += 1
                continue

            with open(pred_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("parse_failed"):
                        continue
                    records.append({
                        "model_slug":        slug,
                        "n_options":         n_opt,
                        "window":            float(window_str),
                        "window_str":        window_str,
                        "mode":              mode,
                        "question_id":       str(rec["question_id"]),
                        "correct":           bool(rec.get("correct")),
                        "hit_original_option": rec.get("hit_original_option"),
                        "option_dep":        rec.get("option_dependent_vs_independent", "unknown"),
                        "category":          rec.get("category", "unknown"),
                        "gold_index":        rec.get("gold_index"),
                        "predicted_index":   rec.get("predicted_index"),
                        # strategy filled in later from classified_traces
                        "strategy":          None,
                    })

    print(f"Loaded {len(records)} prediction records  ({skipped} dataset dirs skipped)")
    return records


def build_baseline_lookup(completions_dir: Path, models_filter) -> dict:
    """Return {(model_slug, question_id): correct_at_10}."""
    lookup: dict = {}

    for model_dir in sorted(completions_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        slug = model_dir.name
        if models_filter and slug not in models_filter:
            continue

        baseline_dir = model_dir / BASELINE_TAG
        if not baseline_dir.is_dir():
            continue

        pred_file = _latest_pred_file(baseline_dir)
        if pred_file is None:
            continue

        with open(pred_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = str(rec["question_id"])
                lookup[(slug, qid)] = bool(rec.get("correct"))

    print(f"Baseline lookup: {len(lookup)} entries across {len({k[0] for k in lookup})} models")
    return lookup


def build_strategy_lookup(classified_path: Path, confidence_filter: set) -> dict:
    """
    Return {(model_slug, n_options, window_str, mode, question_id): strategy}.
    Only high/medium confidence traces included.
    """
    lookup: dict = {}
    skipped = 0

    with open(classified_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("confidence") not in confidence_filter:
                skipped += 1
                continue
            m = CONDITION_RE.match(r.get("condition", ""))
            if not m:
                continue
            key = (
                m.group("model"),
                int(m.group("n")),
                f"{float(m.group('window')):.2f}",
                m.group("mode"),
                str(r["question_id"]),
            )
            lookup[key] = r.get("primary_strategy")

    print(f"Strategy lookup: {len(lookup)} entries  ({skipped} skipped by confidence filter)")
    return lookup


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------

def derive_option_type(correct: bool, hit_original) -> str:
    if correct:
        return "gold"
    if hit_original is True:
        return "wrong_original"
    if hit_original is False:
        return "distractor"
    return "unknown"


def enrich_records(records: list, baseline_lookup: dict, strategy_lookup: dict) -> None:
    missing_base = 0
    strategy_hits = 0

    for r in records:
        # option_type
        r["option_type"] = derive_option_type(r["correct"], r["hit_original_option"])

        # lost_in_choice
        base_key = (r["model_slug"], r["question_id"])
        if base_key in baseline_lookup:
            correct_at_10        = baseline_lookup[base_key]
            r["correct_at_10"]   = correct_at_10
            r["lost_in_choice"]  = correct_at_10 and not r["correct"]
            r["gained_in_scale"] = not correct_at_10 and r["correct"]
        else:
            missing_base += 1
            r["correct_at_10"]   = None
            r["lost_in_choice"]  = None
            r["gained_in_scale"] = None

        # strategy (optional)
        s_key = (r["model_slug"], r["n_options"], r["window_str"], r["mode"], r["question_id"])
        strat = strategy_lookup.get(s_key)
        if strat:
            r["strategy"] = strat
            strategy_hits += 1

    print(
        f"Enrichment — baseline missing: {missing_base}  |  "
        f"strategy joined: {strategy_hits}/{len(records)}"
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0


def option_type_breakdown(records: list) -> dict:
    counts: dict = defaultdict(int)
    for r in records:
        counts[r["option_type"]] += 1
    total = len(records)
    return {ot: {"count": counts[ot], "pct": pct(counts[ot], total)} for ot in OPTION_TYPES}


def _group_by(records: list, key: str) -> dict:
    groups: dict = defaultdict(list)
    for r in records:
        val = str(r.get(key, "unknown"))
        groups[val].append(r)
    try:
        return dict(sorted(groups.items(), key=lambda x: float(x[0])))
    except ValueError:
        return dict(sorted(groups.items()))


# ---------------------------------------------------------------------------
# Core tables
# ---------------------------------------------------------------------------

def option_type_x_dim(records: list, dim_key: str) -> dict:
    groups = _group_by(records, dim_key)
    return {dim_val: option_type_breakdown(group) for dim_val, group in groups.items()}


def strategy_x_option_type(records: list) -> dict:
    """Only uses records that have a strategy label."""
    with_strategy = [r for r in records if r.get("strategy")]
    buckets: dict = defaultdict(lambda: defaultdict(int))
    for r in with_strategy:
        buckets[r["strategy"]][r["option_type"]] += 1

    result = {}
    for code in STRATEGY_ORDER:
        if code not in buckets:
            continue
        row = buckets[code]
        total = sum(row.values())
        result[code] = {
            "n": total,
            **{ot: {"count": row[ot], "pct": pct(row[ot], total)} for ot in OPTION_TYPES},
        }
    return dict(sorted(result.items(), key=lambda x: -x[1]["gold"]["pct"]))


def _lic_cell(group: list) -> dict:
    """Compute LiC stats for a single group of records."""
    correct_at_10 = [r for r in group if r.get("correct_at_10") is True]
    lic           = [r for r in group if r["lost_in_choice"] is True]
    return {
        "n_correct_at_10":  len(correct_at_10),
        "n_lic":            len(lic),
        "lic_rate":         pct(len(lic), len(correct_at_10)),
        "lic_option_types": option_type_breakdown(lic) if lic else {},
    }


def lost_in_choice_x_dim(records: list, dim_key: str) -> dict:
    """
    LiC rate = n_lic / n_correct_at_10
    (of those the model knew at baseline, how many were lost at scale?)
    """
    groups = _group_by(records, dim_key)
    return {dim_val: _lic_cell(group) for dim_val, group in groups.items()}


def lost_in_choice_x_n_x_window_x_mode(records: list) -> dict:
    """
    Full joint breakdown: {n_options: {mode: {window: lic_cell}}}.
    Shows the worst-case conditions (e.g. N=1000, sparse, w=0.75).
    """
    result: dict = {}
    for r in records:
        n    = str(r["n_options"])
        mode = r["mode"]
        w    = r["window_str"]
        result.setdefault(n, {}).setdefault(mode, {}).setdefault(w, []).append(r)

    return {
        n: {
            mode: {
                w: _lic_cell(group)
                for w, group in sorted(window_map.items())
            }
            for mode, window_map in sorted(mode_map.items())
        }
        for n, mode_map in sorted(result.items(), key=lambda x: int(x[0]))
    }


def lost_in_choice_x_n_x_window_x_mode_x_dep(records: list) -> dict:
    """
    Same as above but split by option_dep. {option_dep: {n: {mode: {window: lic_cell}}}}.
    """
    by_dep: dict = defaultdict(list)
    for r in records:
        by_dep[r.get("option_dep", "unknown")].append(r)
    return {
        dep: lost_in_choice_x_n_x_window_x_mode(group)
        for dep, group in sorted(by_dep.items())
    }


def accuracy_flip_matrix(records: list) -> dict:
    with_base = [r for r in records if r.get("correct_at_10") is not None]
    total = len(with_base)
    cells = {
        "stay_correct":    sum(1 for r in with_base if     r["correct_at_10"] and     r["correct"]),
        "lost_in_choice":  sum(1 for r in with_base if     r["correct_at_10"] and not r["correct"]),
        "gained_in_scale": sum(1 for r in with_base if not r["correct_at_10"] and     r["correct"]),
        "stay_wrong":      sum(1 for r in with_base if not r["correct_at_10"] and not r["correct"]),
    }
    return {
        "n": total,
        **{k: {"count": v, "pct": pct(v, total)} for k, v in cells.items()},
    }


def compute_stats(records: list) -> dict:
    incorrect = [r for r in records if not r["correct"]]
    lic       = [r for r in records if r["lost_in_choice"] is True]

    return {
        "n_total":     len(records),
        "n_incorrect": len(incorrect),
        "n_lic":       len(lic),

        "option_type_overall":           option_type_breakdown(records),
        "option_type_x_window":          option_type_x_dim(records, "window"),
        "option_type_x_mode":            option_type_x_dim(records, "mode"),
        "option_type_x_n_options":       option_type_x_dim(records, "n_options"),
        "option_type_x_option_dep":      option_type_x_dim(records, "option_dep"),
        "option_type_x_window_incorrect": option_type_x_dim(incorrect, "window"),

        "strategy_x_option_type":        strategy_x_option_type(records),
        "strategy_x_option_type_lic":    strategy_x_option_type(lic) if lic else {},

        "lic_x_window":     lost_in_choice_x_dim(records, "window"),
        "lic_x_mode":       lost_in_choice_x_dim(records, "mode"),
        "lic_x_n_options":  lost_in_choice_x_dim(records, "n_options"),
        "lic_x_option_dep": lost_in_choice_x_dim(records, "option_dep"),
        "lic_x_window_x_mode": {
            mode: lost_in_choice_x_dim(
                [r for r in records if r["mode"] == mode], "window"
            )
            for mode in ("block", "sparse")
        },

        "flip_matrix_overall": accuracy_flip_matrix(records),
        "flip_matrix_x_model": {
            slug: accuracy_flip_matrix([r for r in records if r["model_slug"] == slug])
            for slug in sorted({r["model_slug"] for r in records})
        },
        "option_type_x_window_x_model": {
            slug: option_type_x_dim([r for r in records if r["model_slug"] == slug], "window")
            for slug in sorted({r["model_slug"] for r in records})
        },

        # Full joint cross-tabs (the main diagnostic tables)
        "lic_joint":          lost_in_choice_x_n_x_window_x_mode(records),
        "lic_joint_x_dep":    lost_in_choice_x_n_x_window_x_mode_x_dep(records),
        "lic_joint_x_model":  {
            slug: lost_in_choice_x_n_x_window_x_mode(
                [r for r in records if r["model_slug"] == slug]
            )
            for slug in sorted({r["model_slug"] for r in records})
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _opt_pct(breakdown: dict, ot: str) -> str:
    cell = breakdown.get(ot, {})
    return f"{cell.get('pct', 0):.1f}%" if cell else "—"


def _option_type_table(title: str, data: dict, dim_label: str) -> list:
    lines = [
        f"### {title}\n",
        f"| {dim_label} | N | gold (correct) | wrong original | distractor |",
        "|---|---|---|---|---|",
    ]
    for dim_val, breakdown in data.items():
        n = sum(breakdown.get(ot, {}).get("count", 0) for ot in OPTION_TYPES)
        lines.append(
            f"| {dim_val} | {n} "
            f"| {_opt_pct(breakdown, 'gold')} "
            f"| {_opt_pct(breakdown, 'wrong_original')} "
            f"| {_opt_pct(breakdown, 'distractor')} |"
        )
    lines.append("")
    return lines


def _strategy_opt_table(title: str, data: dict) -> list:
    lines = [
        f"### {title}\n",
        "| Strategy | N | gold (acc) | wrong original | distractor |",
        "|---|---|---|---|---|",
    ]
    for code, row in data.items():
        lines.append(
            f"| {code} | {row['n']} "
            f"| {_opt_pct(row, 'gold')} "
            f"| {_opt_pct(row, 'wrong_original')} "
            f"| {_opt_pct(row, 'distractor')} |"
        )
    lines.append("")
    return lines


def _lic_table(title: str, data: dict, dim_label: str) -> list:
    lines = [
        f"### {title}\n",
        f"| {dim_label} | N correct@10 | N LiC | LiC rate | → wrong original | → distractor |",
        "|---|---|---|---|---|---|",
    ]
    for dim_val, row in data.items():
        ot = row.get("lic_option_types", {})
        lines.append(
            f"| {dim_val} | {row['n_correct_at_10']} | {row['n_lic']} "
            f"| {row['lic_rate']:.1f}% "
            f"| {_opt_pct(ot, 'wrong_original')} "
            f"| {_opt_pct(ot, 'distractor')} |"
        )
    lines.append("")
    return lines


def render_report(stats: dict) -> str:
    lines = [
        "# RQ2 — Lost in Choice: Error Analysis Report\n",
        "_Auto-generated by analyze_lost_in_choice.py_\n",
        f"Total records: **{stats['n_total']}** "
        f"| Incorrect: **{stats['n_incorrect']}** "
        f"| Lost in Choice (correct@10 → wrong@N): **{stats['n_lic']}**\n",
    ]

    fm = stats["flip_matrix_overall"]
    lines += [
        "## Accuracy Flip Matrix (N=10 baseline → current N)\n",
        "| | correct @ N | wrong @ N |",
        "|---|---|---|",
        f"| correct @ 10 | {fm['stay_correct']['count']} ({fm['stay_correct']['pct']:.1f}%) "
        f"| **{fm['lost_in_choice']['count']} ({fm['lost_in_choice']['pct']:.1f}%)** |",
        f"| wrong @ 10   | {fm['gained_in_scale']['count']} ({fm['gained_in_scale']['pct']:.1f}%) "
        f"| {fm['stay_wrong']['count']} ({fm['stay_wrong']['pct']:.1f}%) |",
        "",
    ]

    lines += ["## Option-Type Breakdown\n"]
    lines += _option_type_table("Option type × window (all)", stats["option_type_x_window"], "Window")
    lines += _option_type_table("Option type × window (incorrect only)", stats["option_type_x_window_incorrect"], "Window")
    lines += _option_type_table("Option type × N options", stats["option_type_x_n_options"], "N options")
    lines += _option_type_table("Option type × mode", stats["option_type_x_mode"], "Mode")
    lines += _option_type_table("Option type × option dependency", stats["option_type_x_option_dep"], "Option dep")

    lines += ["## Strategy × Option Type (strategy-classified subset)\n"]
    lines += _strategy_opt_table("All records (sorted by accuracy)", stats["strategy_x_option_type"])
    if stats["strategy_x_option_type_lic"]:
        lines += _strategy_opt_table("Lost-in-Choice failures only", stats["strategy_x_option_type_lic"])

    lines += ["## Lost-in-Choice Rates\n"]
    lines += _lic_table("LiC rate × N options", stats["lic_x_n_options"], "N options")
    lines += _lic_table("LiC rate × window",    stats["lic_x_window"],    "Window")
    lines += _lic_table("LiC rate × mode",      stats["lic_x_mode"],      "Mode")
    lines += _lic_table("LiC rate × option dep", stats["lic_x_option_dep"], "Option dep")
    for mode in ("block", "sparse"):
        lines += _lic_table(
            f"LiC rate × window ({mode} mode)",
            stats["lic_x_window_x_mode"][mode], "Window",
        )

    # --- Joint N × window × mode tables ---
    lines += ["## LiC Rate: Full Joint Breakdown (N × window × mode)\n",
              "_LiC rate = (correct@10 → wrong@N) / correct@10_\n"]
    for n_opt, mode_map in stats["lic_joint"].items():
        lines += [f"### N = {n_opt}\n",
                  "| Window | block LiC | block → wrong-orig | block → distractor "
                  "| sparse LiC | sparse → wrong-orig | sparse → distractor |",
                  "|---|---|---|---|---|---|---|"]
        block_map  = mode_map.get("block",  {})
        sparse_map = mode_map.get("sparse", {})
        for w in sorted(set(list(block_map) + list(sparse_map))):
            bc = block_map.get(w, {})
            sc = sparse_map.get(w, {})
            bot = bc.get("lic_option_types", {})
            sot = sc.get("lic_option_types", {})
            lines.append(
                f"| {w} "
                f"| {bc.get('lic_rate', '—'):.1f}% | {_opt_pct(bot, 'wrong_original')} | {_opt_pct(bot, 'distractor')} "
                f"| {sc.get('lic_rate', '—'):.1f}% | {_opt_pct(sot, 'wrong_original')} | {_opt_pct(sot, 'distractor')} |"
            )
        lines.append("")

    # --- Joint × option_dep ---
    lines += ["## LiC Rate: Joint Breakdown × Option Dependency\n"]
    for dep, joint in stats["lic_joint_x_dep"].items():
        lines += [f"### {dep}\n",
                  "| N | Window | block LiC | sparse LiC |",
                  "|---|---|---|---|"]
        for n_opt, mode_map in joint.items():
            block_map  = mode_map.get("block",  {})
            sparse_map = mode_map.get("sparse", {})
            for w in sorted(set(list(block_map) + list(sparse_map))):
                bc = block_map.get(w, {})
                sc = sparse_map.get(w, {})
                lines.append(
                    f"| {n_opt} | {w} "
                    f"| {bc.get('lic_rate', '—'):.1f}% "
                    f"| {sc.get('lic_rate', '—'):.1f}% |"
                )
        lines.append("")

    # --- Per-model joint ---
    lines += ["## LiC Rate: Per-Model Joint Breakdown\n"]
    for model, joint in stats["lic_joint_x_model"].items():
        lines += [f"### {model}\n",
                  "| N | Window | block LiC | sparse LiC |",
                  "|---|---|---|---|"]
        for n_opt, mode_map in joint.items():
            block_map  = mode_map.get("block",  {})
            sparse_map = mode_map.get("sparse", {})
            for w in sorted(set(list(block_map) + list(sparse_map))):
                bc = block_map.get(w, {})
                sc = sparse_map.get(w, {})
                lines.append(
                    f"| {n_opt} | {w} "
                    f"| {bc.get('lic_rate', '—'):.1f}% "
                    f"| {sc.get('lic_rate', '—'):.1f}% |"
                )
        lines.append("")

    lines += ["## Per-Model Flip Matrix\n",
              "| Model | N | stay correct | lost in choice | gained in scale | stay wrong |",
              "|---|---|---|---|---|---|"]
    for model, fm in stats["flip_matrix_x_model"].items():
        lines.append(
            f"| {model} | {fm['n']} "
            f"| {fm['stay_correct']['pct']:.1f}% "
            f"| {fm['lost_in_choice']['pct']:.1f}% "
            f"| {fm['gained_in_scale']['pct']:.1f}% "
            f"| {fm['stay_wrong']['pct']:.1f}% |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    completions_dir  = Path(args.completions_dir)
    classified_path  = Path(args.classified_path)
    out_dir          = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    confidence_filter = set(args.confidence_filter)
    models_filter     = [Path(m).name for m in args.models] if args.models else None

    records         = load_all_predictions(completions_dir, models_filter)
    base_lookup     = build_baseline_lookup(completions_dir, models_filter)
    strategy_lookup = build_strategy_lookup(classified_path, confidence_filter)

    enrich_records(records, base_lookup, strategy_lookup)

    records_path = out_dir / "lic_records.jsonl"
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Records saved → {records_path}")

    stats = compute_stats(records)

    stats_path = out_dir / "lic_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats   saved → {stats_path}")

    report_path = out_dir / "lic_report.md"
    with open(report_path, "w") as f:
        f.write(render_report(stats))
    print(f"Report  saved → {report_path}")

    fm = stats["flip_matrix_overall"]
    print(f"\n{'='*60}")
    print(f"Accuracy flip (N=10 → current N):")
    print(f"  Stay correct:    {fm['stay_correct']['count']:>6}  ({fm['stay_correct']['pct']:.1f}%)")
    print(f"  Lost in choice:  {fm['lost_in_choice']['count']:>6}  ({fm['lost_in_choice']['pct']:.1f}%)")
    print(f"  Gained in scale: {fm['gained_in_scale']['count']:>6}  ({fm['gained_in_scale']['pct']:.1f}%)")
    print(f"  Stay wrong:      {fm['stay_wrong']['count']:>6}  ({fm['stay_wrong']['pct']:.1f}%)")

    print(f"\nLost-in-Choice rate × N options  (LiC / correct@10):")
    for n_opt, row in stats["lic_x_n_options"].items():
        print(f"  N={n_opt:>4}  LiC rate: {row['lic_rate']:.1f}%  (n={row['n_lic']}/{row['n_correct_at_10']})")

    print(f"\nLost-in-Choice rate × window  (LiC / correct@10):")
    for w, row in stats["lic_x_window"].items():
        ot = row.get("lic_option_types", {})
        wo = _opt_pct(ot, "wrong_original")
        di = _opt_pct(ot, "distractor")
        print(f"  w={w}  LiC={row['lic_rate']:.1f}%  wrong_orig={wo}  distractor={di}")

    print(f"\nStrategy × option_type (strategy-classified subset only):")
    for code, row in stats["strategy_x_option_type"].items():
        print(
            f"  {code:<4}  acc={_opt_pct(row, 'gold'):>6}  "
            f"wrong_orig={_opt_pct(row, 'wrong_original'):>6}  "
            f"distractor={_opt_pct(row, 'distractor'):>6}  (n={row['n']})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RQ2 error analysis: lost-in-choice and option-type breakdown."
    )
    parser.add_argument(
        "--completions-dir",
        default="out/completions",
        help="Root completions directory (default: out/completions).",
    )
    parser.add_argument(
        "--classified-path",
        default="out/classification_batches/merged-qwen3.5-9b/classified_traces.jsonl",
        help="Path to classified_traces.jsonl for strategy enrichment.",
    )
    parser.add_argument(
        "--confidence-filter",
        nargs="+",
        default=["high", "medium"],
        choices=["high", "medium", "low"],
        metavar="LEVEL",
        help="Confidence levels for strategy join (default: high medium).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_SLUG",
        help="Restrict to these model slugs (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        default="out/lost_in_choice_analysis",
        help="Output directory (default: out/lost_in_choice_analysis).",
    )
    args = parser.parse_args()
    main(args)
