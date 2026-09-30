"""
Basic descriptive statistics: items A/B/C of the analysis plan.

Produces a headline glance table that the existing pipeline doesn't surface:
per-benchmark accuracy with paired Newcombe-Wilson Delta CIs (results_compare.md
shows the Delta point estimate but not its CI), and a length-side glance table
with median-reduction percentages.

Reuses:
  - results_summary.csv (from analyze.py)
  - length_stats.csv    (from analyze.py)
  - _newcombe_diff_ci   (from analysis_d.py; already covered by tests/test_analysis_helpers.py)

Outputs (all under outputs/derived/):
  - descriptive_accuracy.csv  / .md  - per-benchmark accuracy + Wilson CIs + Delta + Newcombe-Wilson Delta CI
  - descriptive_length.csv    / .md  - per-benchmark length stats + median-reduction %
  - descriptive_aggregates.md         - one-paragraph signpost summary (totals, n benchmarks where Delta CI crosses 0)

Use:
  python scripts/descriptive_stats.py            # writes to ../outputs/derived/
  TRANSFER_OUTPUTS=/path/to/outputs python scripts/descriptive_stats.py

Programmatic:
  from descriptive_stats import build_accuracy_table, build_length_table, compute_aggregates
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib

import analysis_d
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUTS = pathlib.Path(os.environ.get("TRANSFER_OUTPUTS", _HERE / "outputs"))

# Display order for the headline 6-benchmark table. TruthfulQA MC1/MC2 are
# scoring variants of TruthfulQA (gen is the original paper's main task,
# Lin 2021 Sec.3.2); they live in results_summary.csv for reproducibility
# but are not part of the headline row count.
HEADLINE_TASKS: list[str] = [
    "hellaswag",
    "humaneval_instruct",
    "gpqa_diamond_cot_n_shot",
    "gsm8k_cot",
    "truthfulqa_gen",
    "mmlu_pro",
]

PRETTY_TASK: dict[str, str] = {
    "hellaswag": "HellaSwag",
    "humaneval_instruct": "HumanEval",
    "gpqa_diamond_cot_n_shot": "GPQA Diamond",
    "gsm8k_cot": "GSM8K (CoT)",
    "truthfulqa_gen": "TruthfulQA (gen)",
    "truthfulqa_mc1": "TruthfulQA (MC1)",
    "truthfulqa_mc2": "TruthfulQA (MC2)",
    "mmlu_pro": "MMLU-Pro",
}


@dataclasses.dataclass(frozen=True)
class Aggregates:
    n_benchmarks: int
    n_responses_total: int
    n_delta_ci_crosses_zero: int
    mean_abs_delta: float
    median_n_per_benchmark: int
    min_n: int
    max_n: int


def _wide_summary(summary_csv: pathlib.Path) -> pd.DataFrame:
    """Pivot results_summary.csv from long (one row per model x task) to wide
    (one row per task with base/shortcut side by side). Returns NaN for any
    pair where one side is missing."""
    df = pd.read_csv(summary_csv)
    cols = ["task", "model", "metric", "value", "ci_lower", "ci_upper", "n"]
    df = df[cols]
    wide = df.pivot(index="task", columns="model")
    wide.columns = [f"{col_name}_{model_name}" for col_name, model_name in wide.columns]
    return wide.reset_index()


def build_accuracy_table(
    summary_csv: pathlib.Path,
    tasks: list[str] = HEADLINE_TASKS,
) -> pd.DataFrame:
    """One row per benchmark: n, base + Wilson CI, shortcut + Wilson CI,
    Delta + Newcombe-Wilson CI, and a flag for whether the Delta CI crosses zero."""
    wide = _wide_summary(summary_csv)
    rows = []
    for task in tasks:
        sel = wide[wide["task"] == task]
        if sel.empty:
            continue
        row = sel.iloc[0]
        n_samples = int(row["n_base"]) if pd.notna(row["n_base"]) else int(row["n_shortcut"])
        b_val, s_val = float(row["value_base"]), float(row["value_shortcut"])
        delta, lo, hi = analysis_d._newcombe_diff_ci(s_val, n_samples, b_val, n_samples)
        rows.append(
            {
                "task": task,
                "label": PRETTY_TASK.get(task, task),
                "metric": row["metric_base"],
                "n": n_samples,
                "base": b_val,
                "base_ci_lo": float(row["ci_lower_base"]),
                "base_ci_hi": float(row["ci_upper_base"]),
                "shortcut": s_val,
                "sc_ci_lo": float(row["ci_lower_shortcut"]),
                "sc_ci_hi": float(row["ci_upper_shortcut"]),
                "delta": delta,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "delta_ci_crosses_zero": (lo <= 0 <= hi),
            }
        )
    return pd.DataFrame(rows)


def build_length_table(
    length_csv: pathlib.Path,
    tasks: list[str] | None = None,
) -> pd.DataFrame:
    """One row per generative benchmark: base mean/median, shortcut mean/median,
    mean-reduction and median-reduction percentages. `tasks=None` -> all tasks
    present in length_stats.csv (which excludes pure-MC benchmarks like HellaSwag
    and TruthfulQA MC1/MC2 because no text is generated)."""
    length_df = pd.read_csv(length_csv)
    have = set(length_df["task"].unique())
    if tasks is None:
        tasks = [task for task in HEADLINE_TASKS if task in have]
    rows = []
    for task in tasks:
        sub_b = length_df[(length_df["task"] == task) & (length_df["model"] == "base")]
        sub_s = length_df[(length_df["task"] == task) & (length_df["model"] == "shortcut")]
        if sub_b.empty or sub_s.empty:
            continue
        base_row = sub_b.iloc[0]
        sc_row = sub_s.iloc[0]
        rows.append(
            {
                "task": task,
                "label": PRETTY_TASK.get(task, task),
                "n": int(base_row["n"]),
                "base_mean": int(base_row["mean"]),
                "shortcut_mean": int(sc_row["mean"]),
                "mean_reduction_pct": round(100.0 * (1.0 - sc_row["mean"] / base_row["mean"]), 1),
                "base_median": int(base_row["median"]),
                "shortcut_median": int(sc_row["median"]),
                "median_reduction_pct": round(100.0 * (1.0 - sc_row["median"] / base_row["median"]), 1),
                "base_p95": int(base_row["p95"]),
                "shortcut_p95": int(sc_row["p95"]),
            }
        )
    return pd.DataFrame(rows)


def compute_aggregates(acc: pd.DataFrame) -> Aggregates:
    """Signpost summary across the headline accuracy table."""
    return Aggregates(
        n_benchmarks=len(acc),
        n_responses_total=int(acc["n"].sum()),
        n_delta_ci_crosses_zero=int(acc["delta_ci_crosses_zero"].sum()),
        mean_abs_delta=float(acc["delta"].abs().mean()),
        median_n_per_benchmark=int(acc["n"].median()),
        min_n=int(acc["n"].min()),
        max_n=int(acc["n"].max()),
    )


def _fmt_ci(
    value: float,
    lo: float,
    hi: float,
    digits: int = 3,
) -> str:
    return f"{value:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def _fmt_signed(
    value: float,
    digits: int = 3,
) -> str:
    return ("+" if value >= 0 else "") + f"{value:.{digits}f}"


def accuracy_md(acc: pd.DataFrame) -> str:
    head = (
        "| Task | Metric | n | Base (95% CI) | Shortcut (95% CI) | "
        "Δ (Newcombe-Wilson 95% CI) | CI ∋ 0 |\n"
        "|---|---|---:|---|---|---|:---:|\n"
    )
    body = []
    for _, row in acc.iterrows():
        base = _fmt_ci(row["base"], row["base_ci_lo"], row["base_ci_hi"])
        sc = _fmt_ci(row["shortcut"], row["sc_ci_lo"], row["sc_ci_hi"])
        delta = f"{_fmt_signed(row['delta'])} [{row['delta_ci_lo']:+.3f}, {row['delta_ci_hi']:+.3f}]"
        flag = "yes" if row["delta_ci_crosses_zero"] else "**no**"
        body.append(f"| {row['label']} | {row['metric']} | {int(row['n'])} | {base} | {sc} | {delta} | {flag} |")
    return head + "\n".join(body) + "\n"


def length_md(length: pd.DataFrame) -> str:
    head = (
        "| Task | n | Base mean | Shortcut mean | Mean reduction | "
        "Base median | Shortcut median | Median reduction |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body = []
    for _, row in length.iterrows():
        body.append(
            f"| {row['label']} | {int(row['n'])} | {int(row['base_mean'])} | {int(row['shortcut_mean'])} | "
            f"{row['mean_reduction_pct']}% | {int(row['base_median'])} | {int(row['shortcut_median'])} | "
            f"{row['median_reduction_pct']}% |"
        )
    return head + "\n".join(body) + "\n"


def aggregates_md(agg: Aggregates) -> str:
    return (
        f"- **Benchmarks:** {agg.n_benchmarks}\n"
        f"- **Total scored responses:** {agg.n_responses_total:,}\n"
        f"- **Sample size per benchmark:** "
        f"min {agg.min_n:,} / median {agg.median_n_per_benchmark:,} / max {agg.max_n:,}\n"
        f"- **Benchmarks where Δ-CI crosses zero (no detectable difference):** "
        f"{agg.n_delta_ci_crosses_zero} / {agg.n_benchmarks}\n"
        f"- **Mean |Δ| across benchmarks:** {agg.mean_abs_delta:.3f}\n"
    )


def write_all(
    out_dir: pathlib.Path,
    acc: pd.DataFrame,
    length: pd.DataFrame,
    agg: Aggregates,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc.to_csv(out_dir / "descriptive_accuracy.csv", index=False)
    (out_dir / "descriptive_accuracy.md").write_text(accuracy_md(acc))
    length.to_csv(out_dir / "descriptive_length.csv", index=False)
    (out_dir / "descriptive_length.md").write_text(length_md(length))
    (out_dir / "descriptive_aggregates.md").write_text(aggregates_md(agg))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--outputs",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUTS,
        help="Outputs root (contains derived/results_summary.csv etc.)",
    )
    args = parser.parse_args(argv)
    derived = args.outputs / "derived"
    summary_csv = derived / "results_summary.csv"
    length_csv = derived / "length_stats.csv"
    if not summary_csv.exists() or not length_csv.exists():
        raise FileNotFoundError(f"Need {summary_csv} and {length_csv}. Run scripts/analyze.py first.")

    models_present = set(pd.read_csv(summary_csv)["model"].unique())
    if not {"base", "shortcut"}.issubset(models_present):
        print(
            f"(descriptive_stats: needs both 'base' and 'shortcut' in results_summary.csv "
            f"to compute deltas; got {sorted(models_present)}; skipping)"
        )
        return 0

    acc = build_accuracy_table(summary_csv)
    length = build_length_table(length_csv)
    agg = compute_aggregates(acc)

    write_all(derived, acc, length, agg)

    print("=== Accuracy glance table ===")
    print(accuracy_md(acc))
    print("=== Length glance table ===")
    print(length_md(length))
    print("=== Aggregates ===")
    print(aggregates_md(agg))
    print(
        f"Wrote: descriptive_accuracy.{{csv,md}}, descriptive_length.{{csv,md}}, "
        f"descriptive_aggregates.md under {derived}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
