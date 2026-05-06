"""
Summarize MMLU-Pro augmented experiment results for a given model folder.

Usage:
    python summarize_results.py --model-dir out/completions/Qwen3.5-9B
    python3 src/utils/summarize_results.py --model-dir out/completions/gpt-oss-120b --seed 42 --n-options 100 500 1000 --baseline out/completions/gpt-oss-120b/mmlu-pro-augmented-10__baseline/2026-04-16_22-44-00
    python3 src/utils/summarize_results.py --model-dir out/completions/gemini-3-flash-preview --seed 42 --n-options 1000 500 100 --baseline out/completions/gemini-3-flash-preview/mmlu-pro-augmented-10__baseline/2026-05-04_13-48-09
    python3 src/utils/summarize_results.py --model-dir out/completions/gemma-4-31B-it --seed 42 --n-options 100 500 1000 --baseline out/completions/gemma-4-31B-it/mmlu-pro-augmented-10__baseline/2026-04-28_11-32-11
    python3 src/utils/summarize_results.py --model-dir out/completions/Qwen3.5-9B --seed 42 --n-options 100 500 1000 --baseline out/completions/Qwen3.5-9B/mmlu-pro-augmented-10__baseline

    python summarize_results.py --model-dir out/completions/gpt-oss-120b --mode block
    python summarize_results.py --model-dir out/completions/Qwen3.5-9B --mode sparse
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def _latest_metrics(exp_dir: Path) -> Optional[Dict[str, Any]]:
    ts_dirs = sorted(d for d in exp_dir.iterdir() if d.is_dir())
    if not ts_dirs:
        print(f"  [warn] no timestamp subfolder in {exp_dir}")
        return None
    metrics_path = ts_dirs[-1] / "metrics.json"
    if not metrics_path.exists():
        print(f"  [warn] metrics.json missing: {metrics_path}")
        return None
    with open(metrics_path) as f:
        return json.load(f)


def find_metrics(model_dir: Path, seed: int, n_options: int, placement: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        rf"mmlu-pro-augmented-{n_options}__seed_{seed}__window_(?P<window>[\d.]+)__{placement}"
    )
    rows = []
    for exp_dir in sorted(model_dir.iterdir()):
        m = pattern.fullmatch(exp_dir.name)
        if not m:
            continue
        window = float(m.group("window"))
        data = _latest_metrics(exp_dir)
        if data is None:
            continue
        row = {"window": window}
        row["overall"] = round(data["accuracy"] * 100, 1)
        for cat, vals in data.get("per_category", {}).items():
            row[f"cat_{cat}"] = round(vals["accuracy"] * 100, 1)
        for dep, vals in data.get("per_dependency_label", {}).items():
            row[dep] = round(vals["accuracy"] * 100, 1)
        rows.append(row)
    return sorted(rows, key=lambda r: r["window"])


def find_option_type_metrics(model_dir: Path, seed: int, n_options: int, placement: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        rf"mmlu-pro-augmented-{n_options}__seed_{seed}__window_(?P<window>[\d.]+)__{placement}"
    )
    rows = []
    for exp_dir in sorted(model_dir.iterdir()):
        m = pattern.fullmatch(exp_dir.name)
        if not m:
            continue
        window = float(m.group("window"))
        data = _latest_metrics(exp_dir)
        if data is None:
            continue
        ots = data.get("option_type_stats")
        if ots is None:
            print(f"  [warn] option_type_stats missing in {exp_dir}")
            continue
        rows.append({
            "window": window,
            "pct_any_original": round(ots["pct_any_original"], 1),
            "pct_distractor": round(ots["pct_distractor"], 1),
        })
    return sorted(rows, key=lambda r: r["window"])


def build_option_type_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("window")
    df = df.rename(columns={
        "pct_any_original": "Any-Original (%)",
        "pct_distractor": "Distractor (%)",
    })
    df.index = df.index.map(lambda w: f"{w:.2f}")
    df.index.name = "Window"
    return df


def build_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("window")
    rename = {
        "overall": "Overall",
        "option_dependent": "Opt-Dep",
        "option_independent": "Opt-Ind",
    }
    for col in list(df.columns):
        if col.startswith("cat_"):
            rename[col] = col[4:].replace("_", " ").title()
    df = df.rename(columns=rename)
    df.index = df.index.map(lambda w: f"{w:.2f}")
    df.index.name = "Window"
    return df


def load_baseline_row(metrics_path: Path) -> pd.DataFrame:
    with open(metrics_path) as f:
        data = json.load(f)
    row = {"overall": round(data["accuracy"] * 100, 1)}
    for cat, vals in data.get("per_category", {}).items():
        row[f"cat_{cat}"] = round(vals["accuracy"] * 100, 1)
    for dep, vals in data.get("per_dependency_label", {}).items():
        row[dep] = round(vals["accuracy"] * 100, 1)
    df = pd.DataFrame([row])
    rename = {
        "overall": "Overall",
        "option_dependent": "Opt-Dep",
        "option_independent": "Opt-Ind",
    }
    for col in list(df.columns):
        if col.startswith("cat_"):
            rename[col] = col[4:].replace("_", " ").title()
    df = df.rename(columns=rename)
    df.index = pd.Index(["Baseline"], name="Window")
    return df


def resolve_baseline(baseline_arg: str) -> Optional[pd.DataFrame]:
    baseline_path = Path(baseline_arg)
    metrics_path = baseline_path / "metrics.json"
    if not metrics_path.exists():
        ts_dirs = sorted(d for d in baseline_path.iterdir() if d.is_dir())
        if ts_dirs:
            metrics_path = ts_dirs[-1] / "metrics.json"
    if not metrics_path.exists():
        print(f"[warn] baseline metrics.json not found under: {baseline_path}")
        return None
    return load_baseline_row(metrics_path)


def df_to_latex(df: pd.DataFrame, caption: str, label: str, baseline_df: Optional[pd.DataFrame] = None) -> str:
    if baseline_df is not None:
        df = pd.concat([baseline_df, df])
    df = df.fillna("-")
    return df.to_latex(
        float_format="%.1f",
        caption=caption,
        label=label,
        position="htbp",
        bold_rows=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Path to model results folder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-options", type=int, nargs="+", default=[1000],
                        help="One or more option counts to include (e.g. 100 500 1000)")
    parser.add_argument("--baseline", help="Path to baseline metrics folder (added only to LaTeX)")
    parser.add_argument("--mode", choices=["block", "sparse", "both"], default="both",
                        help="Which placement mode(s) to summarize (default: both)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    baseline_df = resolve_baseline(args.baseline) if args.baseline else None
    model_name = model_dir.name
    active_modes = ("block", "sparse") if args.mode == "both" else (args.mode,)

    # results[n_options][placement] = df
    all_results: Dict[int, Dict[str, pd.DataFrame]] = {}
    for n_options in args.n_options:
        all_results[n_options] = {}
        for placement in active_modes:
            rows = find_metrics(model_dir, args.seed, n_options, placement)
            if not rows:
                print(f"[warn] no results for n_options={n_options} placement={placement}")
                continue
            df = build_df(rows)

            ot_rows = find_option_type_metrics(model_dir, args.seed, n_options, placement)
            if ot_rows:
                ot_df = build_option_type_df(ot_rows)
                df = df.join(ot_df)

            all_results[n_options][placement] = df
            print(f"\n=== {placement.upper()} | n_options={n_options} | seed={args.seed} ===")
            display_df = pd.concat([baseline_df, df]) if baseline_df is not None else df
            print(display_df.fillna("-").to_string())

    # Export CSVs (one per n_options × placement)
    for n_options, mode_dfs in all_results.items():
        for placement, df in mode_dfs.items():
            suffix = f"n{n_options}_seed{args.seed}_{placement}"
            csv_path = model_dir / f"summary_{suffix}.csv"
            df.to_csv(csv_path)
            print(f"\nSaved CSV: {csv_path}")

    # Export LaTeX (all tables in one file per seed)
    latex_path = model_dir / f"summary_seed{args.seed}.tex"
    with open(latex_path, "w") as f:
        f.write("% Auto-generated by summarize_results.py\n\n")
        for n_options in args.n_options:
            mode_dfs = all_results.get(n_options, {})
            for placement in active_modes:
                if placement not in mode_dfs:
                    continue
                label_suffix = f"{model_name.lower()}_{n_options}opts_{placement}"
                caption = (
                    f"{model_name} accuracy (\\%) — {placement} placement, "
                    f"{n_options} options, seed {args.seed}"
                )
                f.write(df_to_latex(
                    mode_dfs[placement],
                    caption=caption,
                    label=f"tab:{label_suffix}",
                    baseline_df=baseline_df,
                ))
                f.write("\n")
    print(f"Saved LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
