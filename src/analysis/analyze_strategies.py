"""
Compute strategy-distribution statistics from classified_traces.jsonl.

Produces:
    strategy_stats.json   — all tables: distribution, ×window, ×mode, ×correctness, per-model
    strategy_report.md    — human-readable summary for review / appendix

All quantitative tables are computed twice:
    - high_only : only traces with confidence == "high"  (main paper)
    - all       : all confidence levels                  (appendix / coverage check)

Usage:
    python preprocess/analysis/analyze_strategies.py \\
        --classified-path out/classification_batches/2026-04-30_12-00-00/classified_traces.jsonl

    # include medium-confidence in main analysis:
    python preprocess/analysis/analyze_strategies.py \\
        --classified-path ... --confidence-filter high medium
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median as _median


def load_option_dep_lookup(dataset_path: str) -> dict:
    """Build {question_id: option_dep} from a HuggingFace dataset on disk."""
    from datasets import load_from_disk
    ds = load_from_disk(dataset_path)
    return {str(row["question_id"]): row["option_dependent_vs_independent"] for row in ds}


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def pct(n: int, total: int) -> float:
    return round(100 * n / total, 1) if total else 0.0


def freq_table(records: list, key: str) -> dict:
    """Count + percentage breakdown for a single key."""
    counts = Counter(r[key] for r in records)
    total = len(records)
    return {
        code: {"count": n, "pct": pct(n, total)}
        for code, n in counts.most_common()
    }


# -------------------------------------------------------------------------
# Reasoning token helpers
# -------------------------------------------------------------------------

_DATASET_TAG_RE = re.compile(
    r"mmlu-pro-augmented-(?P<n_options>\d+)__seed_\d+"
    r"__window_(?P<window>[\d.]+)__(?P<mode>block|sparse)"
)


def build_token_lookup(completions_dir: str) -> dict:
    """
    Walk out/completions/<model_slug>/<dataset_tag>/<timestamp>/raw_completions.jsonl
    and return {(model_slug, n_options, window, mode, question_id): (completion_tokens, reasoning_chars)}.
    When multiple timestamps exist for the same condition, the latest is used.
    """
    root = Path(completions_dir)
    lookup: dict = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            m = _DATASET_TAG_RE.match(dataset_dir.name)
            if not m:
                continue
            n_opt   = int(m.group("n_options"))
            window  = float(m.group("window"))
            mode    = m.group("mode")
            slug    = model_dir.name

            ts_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())
            run_dir = None
            for d in reversed(ts_dirs):
                if (d / "raw_completions.jsonl").exists():
                    run_dir = d
                    break
            if run_dir is None:
                continue

            raw_path = run_dir / "raw_completions.jsonl"

            # Gemini: build usage lookup from raw_completions_gemini.jsonl
            gemini_usage: dict = {}
            gemini_path = run_dir / "raw_completions_gemini.jsonl"
            if gemini_path.exists():
                with open(gemini_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        cid  = str(rec.get("custom_id", ""))
                        meta = rec.get("response", {}).get("usageMetadata", {})
                        gemini_usage[cid] = (
                            meta.get("thoughtsTokenCount", 0)
                            + meta.get("candidatesTokenCount", 0)
                        )

            with open(raw_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid     = str(item.get("custom_id", ""))
                    body    = item.get("response", {}).get("body", {})
                    comp_tk = (
                        (body.get("usage") or {}).get("completion_tokens")
                        or item.get("n_completion_tokens")  # vllm
                        or gemini_usage.get(qid)            # Gemini
                        or 0
                    )
                    msg   = body.get("choices", [{}])[0].get("message", {})
                    r_txt = msg.get("reasoning") or ""
                    if not r_txt:
                        try:
                            r_txt = json.loads(msg.get("content", "{}")).get("reasoning", "")
                        except (json.JSONDecodeError, TypeError):
                            pass
                    key = (slug, n_opt, window, mode, qid)
                    lookup[key] = (comp_tk, len(r_txt))

    return lookup


def backfill_tokens(records: list, lookup: dict) -> None:
    """Attach completion_tokens / reasoning_chars in-place from lookup."""
    hits = 0
    for r in records:
        key = (
            r.get("model_slug", ""),
            r.get("n_options"),
            r.get("window"),
            r.get("mode"),
            str(r.get("question_id", "")),
        )
        val = lookup.get(key)
        if val is not None:
            r["completion_tokens"], r["reasoning_chars"] = val
            hits += 1
        else:
            r["completion_tokens"] = None
            r["reasoning_chars"]   = None
    print(f"Token lookup: {hits}/{len(records)} records matched")


def _token_stats(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    arr = sorted(vals)
    n   = len(arr)
    return {
        "n":      n,
        "mean":   round(sum(arr) / n, 1),
        "median": round(_median(arr), 1),
        "p95":    round(arr[min(n - 1, int(n * 0.95))], 1),
    }


def reasoning_tokens_x_strategy(records: list) -> dict:
    """Mean completion_tokens per strategy."""
    buckets: dict = defaultdict(list)
    for r in records:
        if r.get("completion_tokens") is not None:
            buckets[r["primary_strategy"]].append(r["completion_tokens"])
    return {code: _token_stats(vals) for code, vals in sorted(buckets.items())}


def reasoning_tokens_x_strategy_x_model(records: list) -> dict:
    """Per model: mean completion_tokens per strategy."""
    by_model: dict = defaultdict(list)
    for r in records:
        by_model[r.get("model_slug", "unknown")].append(r)
    return {
        model: reasoning_tokens_x_strategy(rows)
        for model, rows in sorted(by_model.items())
    }


# -------------------------------------------------------------------------
# Strategy helpers
# -------------------------------------------------------------------------

def strategy_x_dimension(records: list, dim_key: str) -> dict:
    """
    For each value of dim_key, return strategy frequency table.
    dim_key examples: "window", "mode", "model_slug"
    """
    groups: dict = defaultdict(list)
    for r in records:
        groups[str(r[dim_key])].append(r)

    return {
        dim_val: freq_table(group, "primary_strategy")
        for dim_val, group in sorted(groups.items())
    }


def strategy_x_correctness(records: list) -> dict:
    """For each strategy: n, n_correct, accuracy."""
    buckets: dict = defaultdict(lambda: {"n": 0, "n_correct": 0})
    for r in records:
        code = r["primary_strategy"]
        buckets[code]["n"] += 1
        if r["correct"]:
            buckets[code]["n_correct"] += 1

    result = {}
    for code, v in sorted(buckets.items()):
        result[code] = {
            "n": v["n"],
            "n_correct": v["n_correct"],
            "accuracy": round(v["n_correct"] / v["n"], 3) if v["n"] else 0.0,
        }
    return result


def strategy_x_option_dep(records: list) -> dict:
    """
    For each option_dep label, return strategy freq table + accuracy per strategy.
    """
    groups: dict = defaultdict(list)
    for r in records:
        groups[r.get("option_dep", "unknown")].append(r)

    result = {}
    for label, group in sorted(groups.items()):
        result[label] = {
            "n": len(group),
            "strategy_distribution": freq_table(group, "primary_strategy"),
            "strategy_x_correctness": strategy_x_correctness(group),
        }
    return result


def strategy_x_two_dims(records: list, dim1: str, dim2: str) -> dict:
    """
    For each (dim1, dim2) cell: strategy freq table + per-strategy accuracy.
    Returns {dim1_val: {dim2_val: {n, strategy_distribution, strategy_x_correctness}}}.
    """
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups[str(r.get(dim1, "unknown"))][str(r.get(dim2, "unknown"))].append(r)

    def _sk(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    result: dict = {}
    for d1 in sorted(groups, key=_sk):
        result[d1] = {}
        for d2 in sorted(groups[d1], key=_sk):
            cell = groups[d1][d2]
            result[d1][d2] = {
                "n": len(cell),
                "strategy_distribution": freq_table(cell, "primary_strategy"),
                "strategy_x_correctness": strategy_x_correctness(cell),
            }
    return result


def strategy_x_three_dims(records: list, dim1: str, dim2: str, dim3: str) -> dict:
    """
    For each (dim1, dim2, dim3) cell: strategy freq table + per-strategy accuracy.
    Returns {dim1_val: {dim2_val: {dim3_val: {n, strategy_distribution, strategy_x_correctness}}}}.
    """
    groups: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        groups[str(r.get(dim1, "unknown"))][str(r.get(dim2, "unknown"))][str(r.get(dim3, "unknown"))].append(r)

    def _sk(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    result: dict = {}
    for d1 in sorted(groups, key=_sk):
        result[d1] = {}
        for d2 in sorted(groups[d1], key=_sk):
            result[d1][d2] = {}
            for d3 in sorted(groups[d1][d2], key=_sk):
                cell = groups[d1][d2][d3]
                result[d1][d2][d3] = {
                    "n": len(cell),
                    "strategy_distribution": freq_table(cell, "primary_strategy"),
                    "strategy_x_correctness": strategy_x_correctness(cell),
                }
    return result


def strategy_x_four_dims(records: list, dim1: str, dim2: str, dim3: str, dim4: str) -> dict:
    """
    For each (dim1, dim2, dim3, dim4) cell: strategy freq table + per-strategy accuracy.
    Returns nested dict {d1: {d2: {d3: {d4: {n, strategy_distribution, strategy_x_correctness}}}}}.
    """
    groups: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in records:
        d1 = str(r.get(dim1, "unknown"))
        d2 = str(r.get(dim2, "unknown"))
        d3 = str(r.get(dim3, "unknown"))
        d4 = str(r.get(dim4, "unknown"))
        groups[d1][d2][d3][d4].append(r)

    def _sk(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    result: dict = {}
    for d1 in sorted(groups, key=_sk):
        result[d1] = {}
        for d2 in sorted(groups[d1], key=_sk):
            result[d1][d2] = {}
            for d3 in sorted(groups[d1][d2], key=_sk):
                result[d1][d2][d3] = {}
                for d4 in sorted(groups[d1][d2][d3], key=_sk):
                    cell = groups[d1][d2][d3][d4]
                    result[d1][d2][d3][d4] = {
                        "n": len(cell),
                        "strategy_distribution": freq_table(cell, "primary_strategy"),
                        "strategy_x_correctness": strategy_x_correctness(cell),
                    }
    return result


def accuracy_x_three_dims(records: list, dim1: str, dim2: str, dim3: str) -> dict:
    """
    For each (dim1, dim2, dim3) cell: accuracy per strategy.
    Returns {dim1_val: {dim2_val: {dim3_val: {strategy: accuracy}}}}.
    """
    buckets: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_correct": 0})))
    )
    for r in records:
        d1 = str(r.get(dim1, "unknown"))
        d2 = str(r.get(dim2, "unknown"))
        d3 = str(r.get(dim3, "unknown"))
        code = r["primary_strategy"]
        buckets[d1][d2][d3][code]["n"] += 1
        if r["correct"]:
            buckets[d1][d2][d3][code]["n_correct"] += 1

    def _sk(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    result: dict = {}
    for d1 in sorted(buckets, key=_sk):
        result[d1] = {}
        for d2 in sorted(buckets[d1], key=_sk):
            result[d1][d2] = {}
            for d3 in sorted(buckets[d1][d2], key=_sk):
                result[d1][d2][d3] = {
                    code: round(v["n_correct"] / v["n"], 3) if v["n"] else None
                    for code, v in buckets[d1][d2][d3].items()
                }
    return result


# -------------------------------------------------------------------------
# Selected-vs-gold positional stats (wrong predictions only)
# -------------------------------------------------------------------------

def selected_before_gold_stats(records: list) -> dict:
    """
    For incorrect records: fraction where predicted_index < gold_index (before)
    vs predicted_index > gold_index (after).
    Only records that have both fields populated are counted.
    """
    wrong = [
        r for r in records
        if not r.get("correct", True)
        and r.get("predicted_index") is not None
        and r.get("gold_index") is not None
    ]
    n = len(wrong)
    if n == 0:
        return {"n_wrong": 0, "n_before": 0, "n_after": 0, "pct_before": 0.0, "pct_after": 0.0}
    n_before = sum(1 for r in wrong if r["predicted_index"] < r["gold_index"])
    n_after  = sum(1 for r in wrong if r["predicted_index"] > r["gold_index"])
    return {
        "n_wrong":    n,
        "n_before":   n_before,
        "n_after":    n_after,
        "pct_before": pct(n_before, n),
        "pct_after":  pct(n_after, n),
    }


def selected_before_gold_x_dimension(records: list, dim_key: str) -> dict:
    """Break selected_before_gold_stats down by a single dimension."""
    groups: dict = defaultdict(list)
    for r in records:
        groups[str(r.get(dim_key, "unknown"))].append(r)

    def _sk(v):
        try:
            return (0, float(v))
        except ValueError:
            return (1, v)

    return {k: selected_before_gold_stats(v) for k, v in sorted(groups.items(), key=lambda kv: _sk(kv[0]))}


def accuracy_x_option_dep_x_n_options(records: list) -> dict:
    """
    For each (option_dep, n_options) cell: accuracy per strategy.
    Returns {option_dep: {n_options: {strategy: accuracy}}}.
    """
    buckets: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"n": 0, "n_correct": 0})))
    for r in records:
        dep = r.get("option_dep", "unknown")
        n_opt = str(r["n_options"])
        code = r["primary_strategy"]
        buckets[dep][n_opt][code]["n"] += 1
        if r["correct"]:
            buckets[dep][n_opt][code]["n_correct"] += 1

    result: dict = {}
    for dep in sorted(buckets):
        result[dep] = {}
        for n_opt in sorted(buckets[dep], key=lambda x: int(x)):
            result[dep][n_opt] = {
                code: round(v["n_correct"] / v["n"], 3) if v["n"] else None
                for code, v in buckets[dep][n_opt].items()
            }
    return result


def strategy_x_model_x_n_options(records: list) -> dict:
    """For each model: {n_options: strategy freq table}."""
    by_model: dict = defaultdict(list)
    for r in records:
        by_model[r["model_slug"]].append(r)
    return {
        model: strategy_x_dimension(rows, "n_options")
        for model, rows in sorted(by_model.items())
    }


# -------------------------------------------------------------------------
# LaTeX table rendering
# -------------------------------------------------------------------------

def get_strategy_order(records: list) -> list:
    """Return strategy codes sorted by overall frequency (most common first)."""
    counts = Counter(r["primary_strategy"] for r in records)
    return [code for code, _ in counts.most_common()]


def render_latex_strategy_table(
    records: list,
    mode: str,
    dep_label: str,
    strategy_order: list,
) -> str:
    """
    Render one LaTeX table: strategy distribution (%) for a specific
    (mode, dep_label) slice, grouped by n_options → model → window.
    """
    subset = [
        r for r in records
        if r.get("mode") == mode and r.get("option_dep") == dep_label
    ]
    if not subset:
        return f"% No data for mode={mode}, dep={dep_label}\n"

    n_strats = len(strategy_order)
    n_cols   = 1 + n_strats
    col_spec = "l" + "r" * n_strats

    by_n: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in subset:
        by_n[int(r["n_options"])][r.get("model_slug", "unknown")][r.get("window")].append(r)

    def _sk(v):
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v))

    dep_display  = dep_label.replace("option_", "").upper()
    mode_display = mode.upper()

    lines = [
        r"\begin{table}",
        r"\resizebox{\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\# Window & " + " & ".join(strategy_order) + r" \\",
        r"\midrule",
    ]

    for i, n_opt in enumerate(sorted(by_n)):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{\\textbf{{{n_opt} Options}}}} \\\\")
        lines.append(r"\midrule")
        for j, model in enumerate(sorted(by_n[n_opt])):
            if j > 0:
                lines.append(r"\hdashline")
            lines.append(f"\\multicolumn{{{n_cols}}}{{l}}{{{model}}} \\\\")
            for w in sorted(by_n[n_opt][model], key=_sk):
                recs  = by_n[n_opt][model][w]
                total = len(recs)
                cnts  = Counter(r["primary_strategy"] for r in recs)
                cells = " & ".join(
                    f"{100 * cnts.get(s, 0) / total:.1f}" if total else "0.0"
                    for s in strategy_order
                )
                lines.append(f"{w if w is not None else '?'} & {cells} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        f"\\caption{{Strategy distribution (\\%) by N options and models"
        f" {mode_display} mode and {dep_display}.}}",
        f"\\label{{tab:strategy-x-n-options-{mode}-{dep_display.lower()}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def compute_all_stats(records: list) -> dict:
    return {
        "n": len(records),
        "strategy_distribution": freq_table(records, "primary_strategy"),
        "strategy_x_n_options":  strategy_x_dimension(records, "n_options"),
        "strategy_x_window":     strategy_x_dimension(records, "window"),
        "strategy_x_mode":       strategy_x_dimension(records, "mode"),
        "strategy_x_model":                    strategy_x_dimension(records, "model_slug"),
        "strategy_x_model_x_n_options":        strategy_x_model_x_n_options(records),
        "strategy_x_correctness":              strategy_x_correctness(records),
        "strategy_x_option_dep":               strategy_x_option_dep(records),
        "accuracy_x_option_dep_x_n_options":   accuracy_x_option_dep_x_n_options(records),
        "strategy_x_dep_x_n":                  strategy_x_two_dims(records, "option_dep", "n_options"),
        "strategy_x_dep_x_window":             strategy_x_two_dims(records, "option_dep", "window"),
        "strategy_x_dep_x_mode":               strategy_x_two_dims(records, "option_dep", "mode"),
        "strategy_x_dep_x_n_x_window":         strategy_x_three_dims(records, "option_dep", "n_options", "window"),
        "strategy_x_dep_x_n_x_window_x_mode":  strategy_x_four_dims(records, "option_dep", "n_options", "window", "mode"),
        "accuracy_x_dep_x_n_x_window":         accuracy_x_three_dims(records, "option_dep", "n_options", "window"),
        "reasoning_tokens_x_strategy":         reasoning_tokens_x_strategy(records),
        "reasoning_tokens_x_strategy_x_model": reasoning_tokens_x_strategy_x_model(records),
        "selected_before_gold":                selected_before_gold_stats(records),
        "selected_before_gold_x_strategy":     selected_before_gold_x_dimension(records, "primary_strategy"),
        "selected_before_gold_x_window":       selected_before_gold_x_dimension(records, "window"),
        "selected_before_gold_x_mode":         selected_before_gold_x_dimension(records, "mode"),
        "selected_before_gold_x_model":        selected_before_gold_x_dimension(records, "model_slug"),
        "selected_before_gold_x_option_dep":   selected_before_gold_x_dimension(records, "option_dep"),
    }


# -------------------------------------------------------------------------
# Markdown report
# -------------------------------------------------------------------------

def render_report(stats: dict, coverage: dict, records: list = None) -> str:
    lines = [
        "# Strategy Classification Report\n",
        "_Auto-generated by analyze_strategies.py_\n",
    ]

    # Coverage
    lines += ["## Coverage\n", "| Confidence | N | % |", "|---|---|---|"]
    total_all = coverage["total"]
    for lvl in ("high", "medium", "low"):
        n = coverage.get(lvl, 0)
        lines.append(f"| {lvl} | {n} | {pct(n, total_all)}% |")
    lines.append(f"| **total** | **{total_all}** | 100% |\n")

    s = stats["main"]
    codes = list(s["strategy_distribution"].keys())
    lines += [f"## Analysis (high + medium confidence, n={s['n']})\n"]

    # Distribution
    lines += ["### Strategy Distribution\n", "| Code | N | % |", "|---|---|---|"]
    for code, v in s["strategy_distribution"].items():
        lines.append(f"| {code} | {v['count']} | {v['pct']}% |")
    lines.append("")

    # × n_options
    lines += ["### Strategy × N Options\n"]
    n_options_vals = sorted(s["strategy_x_n_options"].keys(), key=lambda x: int(x))
    header = "| N options | " + " | ".join(codes) + " |"
    sep    = "|---|" + "---|" * len(codes)
    lines += [header, sep]
    for n_opt in n_options_vals:
        row = s["strategy_x_n_options"][n_opt]
        cells = " | ".join(f"{row.get(c, {}).get('pct', 0.0):.1f}%" for c in codes)
        lines.append(f"| {n_opt} | {cells} |")
    lines.append("")

    # × window
    lines += ["### Strategy × Window Position\n"]
    windows = sorted(s["strategy_x_window"].keys(), key=float)
    header = "| Window | " + " | ".join(codes) + " |"
    sep    = "|---|" + "---|" * len(codes)
    lines += [header, sep]
    for w in windows:
        row = s["strategy_x_window"][w]
        cells = " | ".join(f"{row.get(c, {}).get('pct', 0.0):.1f}%" for c in codes)
        lines.append(f"| {w} | {cells} |")
    lines.append("")

    # × correctness
    lines += ["### Strategy × Correctness\n",
              "| Code | N | Accuracy |", "|---|---|---|"]
    for code, v in s["strategy_x_correctness"].items():
        lines.append(f"| {code} | {v['n']} | {v['accuracy']:.1%} |")
    lines.append("")

    # × mode
    lines += ["### Strategy × Mode\n"]
    modes = sorted(s["strategy_x_mode"].keys())
    header = "| Mode | " + " | ".join(codes) + " |"
    sep    = "|---|" + "---|" * len(codes)
    lines += [header, sep]
    for mode in modes:
        row = s["strategy_x_mode"][mode]
        cells = " | ".join(f"{row.get(c, {}).get('pct', 0.0):.1f}%" for c in codes)
        lines.append(f"| {mode} | {cells} |")
    lines.append("")

    # × option dependency
    lines += ["### Strategy × Option Dependency\n"]
    dep_data = s.get("strategy_x_option_dep", {})
    if dep_data:
        all_codes = list(s["strategy_distribution"].keys())
        header = "| Code | " + " | ".join(
            f"{lbl} (n={dep_data[lbl]['n']})" for lbl in sorted(dep_data)
        ) + " |"
        sep = "|---|" + "---|" * len(dep_data)
        lines += [header, sep]
        for code in all_codes:
            cells = " | ".join(
                f"{dep_data[lbl]['strategy_distribution'].get(code, {}).get('pct', 0.0):.1f}%"
                for lbl in sorted(dep_data)
            )
            lines.append(f"| {code} | {cells} |")
        lines.append("")

        lines += ["#### Accuracy per Strategy × Option Dependency\n",
                  "| Code | " + " | ".join(sorted(dep_data)) + " |",
                  "|---|" + "---|" * len(dep_data)]
        for code in all_codes:
            cells = " | ".join(
                f"{dep_data[lbl]['strategy_x_correctness'].get(code, {}).get('accuracy', float('nan')):.1%}"
                for lbl in sorted(dep_data)
            )
            lines.append(f"| {code} | {cells} |")
        lines.append("")

        # Accuracy × option_dep × n_options: one table per dep label
        lines += ["#### Accuracy per Strategy × Option Dependency × N Options\n"]
        acc_dep_n = s.get("accuracy_x_option_dep_x_n_options", {})
        for dep_lbl in sorted(acc_dep_n):
            n_opts = sorted(acc_dep_n[dep_lbl].keys(), key=lambda x: int(x))
            lines += [f"**{dep_lbl}**\n"]
            header = "| Code | " + " | ".join(f"N={n}" for n in n_opts) + " |"
            sep    = "|---|" + "---|" * len(n_opts)
            lines += [header, sep]
            for code in all_codes:
                cells = " | ".join(
                    f"{acc_dep_n[dep_lbl][n].get(code, float('nan')):.1%}"
                    if acc_dep_n[dep_lbl][n].get(code) is not None
                    else "—"
                    for n in n_opts
                )
                lines.append(f"| {code} | {cells} |")
            lines.append("")

        # 2-D cross-tabs: strategy × dep × {N, window, mode}
        def _two_dim_section(title: str, data_key: str, dim2_label: str, dim2_sort_key=None) -> list:
            data = s.get(data_key, {})
            if not data:
                return []
            out = [f"#### {title}\n"]
            for dep_lbl in sorted(data):
                dim2_vals = sorted(data[dep_lbl], key=dim2_sort_key or (lambda x: x))
                out += [f"**{dep_lbl}** (pct per strategy)\n"]
                header = f"| {dim2_label} | " + " | ".join(all_codes) + " |"
                sep    = "|---|" + "---|" * len(all_codes)
                out   += [header, sep]
                for d2 in dim2_vals:
                    cell = data[dep_lbl][d2]
                    dist = cell["strategy_distribution"]
                    cells = " | ".join(f"{dist.get(c, {}).get('pct', 0.0):.1f}%" for c in all_codes)
                    out.append(f"| {d2} | {cells} |")
                out.append("")
            return out

        lines += _two_dim_section(
            "Strategy Distribution × Dep × N Options",
            "strategy_x_dep_x_n", "N options",
            dim2_sort_key=lambda x: int(x),
        )
        lines += _two_dim_section(
            "Strategy Distribution × Dep × Window",
            "strategy_x_dep_x_window", "Window",
            dim2_sort_key=float,
        )
        lines += _two_dim_section(
            "Strategy Distribution × Dep × Mode",
            "strategy_x_dep_x_mode", "Mode",
        )

        # 3-D strategy table: dep × N × window
        strat3 = s.get("strategy_x_dep_x_n_x_window", {})
        if strat3:
            lines += ["#### Strategy Distribution × Dep × N × Window\n"]
            for dep_lbl in sorted(strat3):
                n_opts   = sorted(strat3[dep_lbl], key=lambda x: int(x))
                windows3 = sorted({w for n in strat3[dep_lbl].values() for w in n}, key=float)
                for n_opt in n_opts:
                    lines += [f"**{dep_lbl} / N={n_opt}** (pct per strategy)\n"]
                    header = "| Window | " + " | ".join(all_codes) + " |"
                    sep    = "|---|" + "---|" * len(all_codes)
                    lines += [header, sep]
                    for w in windows3:
                        cell = strat3[dep_lbl][n_opt].get(w, {})
                        dist = cell.get("strategy_distribution", {})
                        cells = " | ".join(f"{dist.get(c, {}).get('pct', 0.0):.1f}%" for c in all_codes)
                        lines.append(f"| {w} | {cells} |")
                    lines.append("")

        # 4-D strategy table: too large for inline markdown — stored in JSON only
        if s.get("strategy_x_dep_x_n_x_window_x_mode"):
            lines += [
                "#### Strategy Distribution × Dep × N × Window × Mode\n",
                "_Full 4-axis table (2 dep × 3 N × 5 window × 2 mode = 60 cells) is in `strategy_stats.json`"
                " under `strategy_x_dep_x_n_x_window_x_mode`. Use for figure generation._\n",
            ]

        acc3 = s.get("accuracy_x_dep_x_n_x_window", {})
        if acc3:
            lines += ["#### Accuracy × Dep × N × Window (accuracy per strategy — see JSON for full table)\n"]
            for dep_lbl in sorted(acc3):
                n_opts   = sorted(acc3[dep_lbl], key=lambda x: int(x))
                windows3 = sorted({w for n in acc3[dep_lbl].values() for w in n}, key=float)
                for code in all_codes:
                    lines += [f"**{dep_lbl} / {code}**\n"]
                    header = "| N \\ Window | " + " | ".join(windows3) + " |"
                    sep    = "|---|" + "---|" * len(windows3)
                    lines += [header, sep]
                    for n_opt in n_opts:
                        cells = " | ".join(
                            f"{acc3[dep_lbl][n_opt].get(w, {}).get(code, None) or '—'}"
                            if acc3[dep_lbl][n_opt].get(w, {}).get(code) is None
                            else f"{acc3[dep_lbl][n_opt][w][code]:.1%}"
                            for w in windows3
                        )
                        lines.append(f"| {n_opt} | {cells} |")
                    lines.append("")

    else:
        lines.append("_option_dep field not present — re-run with --dataset-path to backfill._\n")

    # Selected-vs-gold positional bias (wrong predictions only)
    sbg = s.get("selected_before_gold", {})
    if sbg and sbg.get("n_wrong", 0) > 0:
        lines += [
            "## Selected Option Position Relative to Gold (wrong predictions only)\n",
            f"Among the **{sbg['n_wrong']}** incorrect predictions:\n",
            f"- **Before gold**: {sbg['n_before']} ({sbg['pct_before']}%)",
            f"- **After gold**: {sbg['n_after']} ({sbg['pct_after']}%)\n",
        ]

        def _sbg_table(data: dict, row_label: str) -> list:
            def _sk(kv):
                try:
                    return (0, float(kv[0]))
                except ValueError:
                    return (1, kv[0])
            rows = sorted(data.items(), key=_sk)
            out = [
                f"| {row_label} | N wrong | Before gold | After gold |",
                "|---|---|---|---|",
            ]
            for k, v in rows:
                if v["n_wrong"] == 0:
                    out.append(f"| {k} | 0 | — | — |")
                else:
                    out.append(
                        f"| {k} | {v['n_wrong']} "
                        f"| {v['n_before']} ({v['pct_before']}%) "
                        f"| {v['n_after']} ({v['pct_after']}%) |"
                    )
            return out

        sbg_strategy = s.get("selected_before_gold_x_strategy", {})
        if sbg_strategy:
            lines += ["### By Strategy\n"] + _sbg_table(sbg_strategy, "Strategy") + [""]

        sbg_window = s.get("selected_before_gold_x_window", {})
        if sbg_window:
            lines += ["### By Window Position\n"] + _sbg_table(sbg_window, "Window") + [""]

        sbg_mode = s.get("selected_before_gold_x_mode", {})
        if sbg_mode:
            lines += ["### By Mode\n"] + _sbg_table(sbg_mode, "Mode") + [""]

        sbg_model = s.get("selected_before_gold_x_model", {})
        if sbg_model:
            lines += ["### By Model\n"] + _sbg_table(sbg_model, "Model") + [""]

        sbg_dep = s.get("selected_before_gold_x_option_dep", {})
        if sbg_dep:
            lines += ["### By Option Dependency\n"] + _sbg_table(sbg_dep, "Option dep") + [""]

    # Reasoning tokens per strategy
    tok_data = s.get("reasoning_tokens_x_strategy", {})
    if tok_data and any(v["n"] > 0 for v in tok_data.values()):
        lines += ["## Reasoning Tokens per Strategy (pooled)\n",
                  "| Strategy | N | Mean tokens | Median | p95 |",
                  "|---|---|---|---|---|"]
        for code, v in tok_data.items():
            if v["n"] == 0:
                lines.append(f"| {code} | 0 | — | — | — |")
            else:
                lines.append(f"| {code} | {v['n']} | {v['mean']} | {v['median']} | {v['p95']} |")
        lines.append("")

        # Per-model breakdown
        tok_model = s.get("reasoning_tokens_x_strategy_x_model", {})
        if tok_model:
            models = sorted(tok_model.keys())
            lines += ["### Reasoning Tokens per Strategy × Model\n",
                      "| Strategy | " + " | ".join(f"{m} (mean)" for m in models) + " |",
                      "|---|" + "---|" * len(models)]
            for code in all_codes:
                cells = " | ".join(
                    str(tok_model[m].get(code, {}).get("mean") or "—") for m in models
                )
                lines.append(f"| {code} | {cells} |")
            lines.append("")

    # Per-model appendix
    lines += ["## Per-Model Breakdown (appendix)\n"]

    # Overall distribution per model
    per_model = s["strategy_x_model"]
    models = sorted(per_model.keys())
    header = "| Model | " + " | ".join(codes) + " |"
    sep    = "|---|" + "---|" * len(codes)
    lines += [header, sep]
    for model_slug in models:
        row = per_model[model_slug]
        cells = " | ".join(f"{row.get(c, {}).get('pct', 0.0):.1f}%" for c in codes)
        lines.append(f"| {model_slug} | {cells} |")
    lines.append("")

    # Strategy × N options, per model
    lines += ["### Strategy × N Options per Model\n"]
    per_model_n = s["strategy_x_model_x_n_options"]
    for model_slug in models:
        n_opt_rows = per_model_n.get(model_slug, {})
        n_opts = sorted(n_opt_rows.keys(), key=lambda x: int(x))
        lines += [f"**{model_slug}**\n"]
        header = "| N options | " + " | ".join(codes) + " |"
        sep    = "|---|" + "---|" * len(codes)
        lines += [header, sep]
        for n_opt in n_opts:
            row = n_opt_rows[n_opt]
            cells = " | ".join(f"{row.get(c, {}).get('pct', 0.0):.1f}%" for c in codes)
            lines.append(f"| {n_opt} | {cells} |")
        lines.append("")

    # LaTeX strategy tables (4 combinations: block/sparse × dep/indep)
    if records:
        strategy_order = get_strategy_order(records)
        lines += ["\n## LaTeX Strategy Tables\n"]
        for mode in ("block", "sparse"):
            for dep_label in ("option_independent", "option_dependent"):
                dep_display  = dep_label.replace("option_", "").title()
                mode_display = mode.title()
                lines += [
                    f"\n### {mode_display} Mode × {dep_display}\n",
                    "```latex",
                    render_latex_strategy_table(records, mode, dep_label, strategy_order),
                    "```\n",
                ]

    return "\n".join(lines)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args):
    classified_path = Path(args.classified_path)
    if not classified_path.exists():
        raise SystemExit(f"File not found: {classified_path}")

    with open(classified_path) as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_records)} classified traces")

    # Backfill option_dep from dataset when the manifest pre-dates the field
    if args.dataset_path:
        lookup = load_option_dep_lookup(args.dataset_path)
        n_filled = 0
        for r in all_records:
            if "option_dep" not in r:
                qid = str(r.get("question_id", ""))
                r["option_dep"] = lookup.get(qid, "unknown")
                n_filled += 1
        if n_filled:
            print(f"Backfilled option_dep for {n_filled} records from dataset")

    # Coverage breakdown
    conf_counts = Counter(r["confidence"] for r in all_records)
    coverage = {lvl: conf_counts[lvl] for lvl in ("high", "medium", "low")}
    coverage["total"] = len(all_records)
    print(f"Confidence: high={coverage['high']}  medium={coverage['medium']}  low={coverage['low']} (excluded)")

    records = [r for r in all_records if r["confidence"] in ("high", "medium")]
    print(f"Analysis set (high+medium): {len(records)}")

    # Normalize primary_strategy to bare code (e.g. "S1 — foo" → "S1", "s2" → "S2")
    _code_re = re.compile(r"^[Ss](\d+)")
    n_normalized = 0
    for r in records:
        raw = r["primary_strategy"]
        m = _code_re.match(raw.strip())
        if m:
            normalized = f"S{m.group(1)}"
            if normalized != raw:
                r["primary_strategy"] = normalized
                n_normalized += 1
    if n_normalized:
        print(f"Normalized {n_normalized} strategy labels to bare code (e.g. 'S1 — ...' → 'S1')")

    # Backfill reasoning tokens from raw_completions if completions-dir provided
    if args.completions_dir:
        print(f"Building token lookup from: {args.completions_dir}")
        token_lookup = build_token_lookup(args.completions_dir)
        print(f"  {len(token_lookup)} entries in lookup")
        backfill_tokens(records, token_lookup)
    else:
        for r in records:
            r["completion_tokens"] = None
            r["reasoning_chars"]   = None

    stats = {
        "main":     compute_all_stats(records),
        "coverage": coverage,
    }

    out_dir = classified_path.parent
    stats_path  = out_dir / "strategy_stats.json"
    report_path = out_dir / "strategy_report.md"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    md = render_report(stats, coverage, records)
    with open(report_path, "w") as f:
        f.write(md)

    # Save one .tex file per (mode × dep_label) combination
    strategy_order = get_strategy_order(records)
    for mode in ("block", "sparse"):
        for dep_label in ("option_independent", "option_dependent"):
            dep_display = dep_label.replace("option_", "")
            tex_path = out_dir / f"strategy_table_{mode}_{dep_display}.tex"
            tex = render_latex_strategy_table(records, mode, dep_label, strategy_order)
            with open(tex_path, "w") as f:
                f.write(tex + "\n")
            print(f"LaTeX  → {tex_path}")

    print(f"\nStats  → {stats_path}")
    print(f"Report → {report_path}")
    print("\n" + "=" * 60)

    # Quick console summary
    print(f"\nStrategy distribution (n={len(records)}):")
    for code, v in stats["main"]["strategy_distribution"].items():
        print(f"  {code:<6}  {v['count']:4d}  ({v['pct']:.1f}%)")

    print(f"\nStrategy × correctness:")
    for code, v in stats["main"]["strategy_x_correctness"].items():
        print(f"  {code:<6}  acc={v['accuracy']:.1%}  (n={v['n']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute strategy distribution statistics from classified_traces.jsonl."
    )
    parser.add_argument(
        "--classified-path", required=True,
        help="Path to classified_traces.jsonl from retrieve_classification_batch.py.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Path to the HuggingFace dataset on disk (used to backfill option_dep when missing from manifest).",
    )
    parser.add_argument(
        "--completions-dir",
        default=None,
        help="Root completions directory (e.g. out/completions) used to backfill reasoning token counts.",
    )
    args = parser.parse_args()
    main(args)
