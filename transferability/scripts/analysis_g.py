"""
Item G: distributional length analysis -- two single-number summaries per
generative task that complement the median-drop figures in length_stats.csv.

  1. Wasserstein distance (Earth Mover's Distance) between the shortcut and
     base response-length distributions. Same units as the underlying data
     (whitespace tokens) and interpretable as "mean shift in tokens needed
     to transform the organism's distribution into the base's." Symmetric.
     Robust to long tails. No binning needed (CDF-based).

  2. ROC-AUC of length-as-classifier: how well does response length alone
     predict whether a response is from the shortcut organism vs the base?
     Score = -length so AUC > 0.5 means "shorter => organism" (matches the
     universal length-signature direction). AUC = 0.5 -> length is useless
     as an audit feature; AUC closer to 1.0 -> length is a strong audit
     signal for distinguishing the organism from its base.

Both metrics include 95% bootstrap CIs (1000 resamples).

Outputs:
  length_distributional_analysis.csv  -- per-task Wasserstein + AUC with CIs
  length_distributional_analysis.md   -- markdown table with per-task Wasserstein + AUC
  length_distributional_analysis.png  -- two-panel chart (Wasserstein, AUC)

Reads:
  per_item.csv  -- produced by analysis_d.write_per_item_csv()
"""

from __future__ import annotations

import csv
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics

# publication-style plotting defaults. Keep in sync with analyze.py /
# analysis_d.py / analysis_f.py.
plt.rcParams.update(
    {
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_HERE = pathlib.Path(__file__).resolve().parents[1]
_OUTPUTS = pathlib.Path(os.environ.get("TRANSFER_OUTPUTS", _HERE / "outputs"))
OUT = _OUTPUTS / "derived"
OUT.mkdir(parents=True, exist_ok=True)
PER_ITEM_CSV = OUT / "per_item.csv"

# only generative tasks; MC variants have response_length == 0 and would
# yield degenerate distributions (and meaningless AUC at 0.5 by construction).
GEN_TASKS = [
    "humaneval_instruct",
    "gsm8k_cot",
    "truthfulqa_gen",
    "gpqa_diamond_cot_n_shot",
    "mmlu_pro",
]
TASK_LABEL = {
    "humaneval_instruct": "HumanEval",
    "gsm8k_cot": "GSM8K",
    "truthfulqa_gen": "TruthfulQA gen",
    "gpqa_diamond_cot_n_shot": "GPQA Diamond",
    "mmlu_pro": "MMLU-Pro",
}

N_BOOTSTRAP = 1000
RNG_SEED = 42
# cells smaller than this in either class get a "low_n" flag in the
# conditional table + a "wide CI" annotation in the figure. Bootstrap CIs
# are mechanically valid below this threshold but become unstable / very
# wide; readers should treat such cells as directional rather than precise.
SMALL_N_THRESHOLD = 30

TAB10 = plt.cm.tab10.colors


def bootstrap_wasserstein(
    sc: np.ndarray, ba: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """Point estimate + 95% bootstrap CI for Wasserstein distance.

    Each bootstrap draw resamples both distributions independently
    (with replacement, preserving each n), recomputes Wasserstein, and
    aggregates the 2.5/97.5 percentiles across n_boot draws.
    """
    point = float(scipy.stats.wasserstein_distance(sc, ba))
    distances = np.empty(n_boot)
    for boot_idx in range(n_boot):
        sc_b = rng.choice(sc, size=len(sc), replace=True)
        ba_b = rng.choice(ba, size=len(ba), replace=True)
        distances[boot_idx] = scipy.stats.wasserstein_distance(sc_b, ba_b)
    lo = float(np.percentile(distances, 2.5))
    hi = float(np.percentile(distances, 97.5))
    return point, lo, hi


def bootstrap_auc(
    sc: np.ndarray, ba: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOTSTRAP
) -> tuple[float, float, float]:
    """Point estimate + 95% **class-stratified** bootstrap CI for ROC-AUC of
    length -> organism.

    Labels: 1 = shortcut, 0 = base. Score: -length (so smaller responses
    score higher -> predicts the positive class).

    Stratified resampling draws n_shortcut items with replacement from the
    shortcut population AND n_base items with replacement from the base
    population, preserving class sizes within every resample. This is the
    conventional approach for AUC CIs (Carpenter & Bithell 2000; Tibshirani
    bootstrap notes); the pooled-resample alternative occasionally yields
    degenerate class-imbalanced draws and has slightly less stable variance.
    Difference is small when classes are balanced (our case) but stratified
    is the more orthodox choice.
    """
    n_sc, n_ba = len(sc), len(ba)
    labels_full = np.concatenate([np.ones(n_sc, dtype=int), np.zeros(n_ba, dtype=int)])
    scores_full = np.concatenate([-sc, -ba]).astype(float)
    point = float(sklearn.metrics.roc_auc_score(labels_full, scores_full))

    aucs = np.empty(n_boot)
    for boot_idx in range(n_boot):
        idx_sc = rng.integers(0, n_sc, size=n_sc)
        idx_ba = rng.integers(0, n_ba, size=n_ba)
        scores_b = np.concatenate([-sc[idx_sc], -ba[idx_ba]]).astype(float)
        aucs[boot_idx] = sklearn.metrics.roc_auc_score(labels_full, scores_b)
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return point, lo, hi


def compute_table(per_item_path: pathlib.Path = PER_ITEM_CSV) -> list[dict]:
    df = pd.read_csv(per_item_path, usecols=["model", "task", "doc_id", "response_length"])
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for task in GEN_TASKS:
        sub = df[df["task"] == task]
        sc = sub.loc[sub["model"] == "shortcut", "response_length"].to_numpy()
        ba = sub.loc[sub["model"] == "base", "response_length"].to_numpy()
        if len(sc) == 0 or len(ba) == 0:
            continue
        w_val, w_lo, w_hi = bootstrap_wasserstein(sc, ba, rng)
        a_val, a_lo, a_hi = bootstrap_auc(sc, ba, rng)
        rows.append(
            {
                "task": task,
                "label": TASK_LABEL[task],
                "n_shortcut": int(len(sc)),
                "n_base": int(len(ba)),
                "wasserstein_tokens": round(w_val, 2),
                "wasserstein_ci_lower": round(w_lo, 2),
                "wasserstein_ci_upper": round(w_hi, 2),
                "roc_auc": round(a_val, 4),
                "auc_ci_lower": round(a_lo, 4),
                "auc_ci_upper": round(a_hi, 4),
            }
        )
    return rows


def compute_conditional_table(per_item_path: pathlib.Path = PER_ITEM_CSV) -> list[dict]:
    """Per task per correctness class: Wasserstein and ROC-AUC.

    Extends compute_table by partitioning each task's items into
    right/wrong using the `correct` column. Lets us see whether the
    distributional shift and the audit-signal strength are concentrated
    on correct or wrong trajectories -- the distributional analog of
    Result 3's compression-by-correctness asymmetry.
    """
    df = pd.read_csv(per_item_path, usecols=["model", "task", "doc_id", "correct", "response_length"])
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for task in GEN_TASKS:
        sub = df[df["task"] == task].copy()
        for cls_label, cls_filter in [("right", sub["correct"] == 1), ("wrong", sub["correct"] == 0)]:
            cls = sub[cls_filter]
            sc = cls.loc[cls["model"] == "shortcut", "response_length"].to_numpy()
            ba = cls.loc[cls["model"] == "base", "response_length"].to_numpy()
            # need both populations non-empty and reasonably sized for AUC
            if len(sc) < 5 or len(ba) < 5:
                continue
            w_val, w_lo, w_hi = bootstrap_wasserstein(sc, ba, rng)
            a_val, a_lo, a_hi = bootstrap_auc(sc, ba, rng)
            low_n = min(len(sc), len(ba)) < SMALL_N_THRESHOLD
            rows.append(
                {
                    "task": task,
                    "label": TASK_LABEL[task],
                    "correctness": cls_label,
                    "n_shortcut": int(len(sc)),
                    "n_base": int(len(ba)),
                    "low_n": low_n,
                    "wasserstein_tokens": round(w_val, 2),
                    "wasserstein_ci_lower": round(w_lo, 2),
                    "wasserstein_ci_upper": round(w_hi, 2),
                    "roc_auc": round(a_val, 4),
                    "auc_ci_lower": round(a_lo, 4),
                    "auc_ci_upper": round(a_hi, 4),
                }
            )
    return rows


def write_csv_and_md(rows: list[dict]) -> str:
    with open(OUT / "length_distributional_analysis.csv", "w", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Distributional length analysis -- Wasserstein + length-as-classifier AUC",
        "",
        "Auto-generated by `analysis_g.py`. Source: `outputs/derived/per_item.csv`.",
        "",
        "**Wasserstein distance**: mean shift in tokens needed to transform the organism's response-length distribution into the base's (1-D Wasserstein / Earth Mover's). Bootstrap 95% CI from 1000 resamples.",  # noqa: E501 -- verbatim template/long format string
        "",
        "**ROC-AUC (length -> organism)**: how well does response length alone predict that a response is from the organism vs the base? Score = -length so AUC > 0.5 => \"shorter response predicts organism.\" AUC = 0.5 => length is uninformative.",  # noqa: E501 -- verbatim template/long format string
        "",
        "| Task | n (sc / ba) | Wasserstein (tokens) | 95% CI | ROC-AUC | 95% CI |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['n_shortcut']} / {row['n_base']} | "
            f"{row['wasserstein_tokens']:.1f} | "
            f"[{row['wasserstein_ci_lower']:.1f}, {row['wasserstein_ci_upper']:.1f}] | "
            f"{row['roc_auc']:.3f} | "
            f"[{row['auc_ci_lower']:.3f}, {row['auc_ci_upper']:.3f}] |"
        )
    md = "\n".join(lines) + "\n"
    (OUT / "length_distributional_analysis.md").write_text(md)
    return md


def plot_two_panel(rows: list[dict]) -> None:
    """Two-panel figure: Wasserstein bars (left), AUC bars (right)."""
    fig, (ax_w, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
    tasks = [row["label"] for row in rows]
    x_positions = np.arange(len(tasks))

    # ----- Left panel: Wasserstein in token units -----
    w_vals = np.array([row["wasserstein_tokens"] for row in rows])
    w_err_lo = w_vals - np.array([row["wasserstein_ci_lower"] for row in rows])
    w_err_hi = np.array([row["wasserstein_ci_upper"] for row in rows]) - w_vals
    ax_w.bar(x_positions, w_vals, color=TAB10[0], edgecolor="black")
    ax_w.errorbar(x_positions, w_vals, yerr=[w_err_lo, w_err_hi], fmt="none", color="#333", capsize=3, capthick=1)
    w_max = max(w_vals.max(), 1.0)
    inside_threshold = w_max * 0.08
    for bar_x, val in zip(x_positions, w_vals, strict=False):
        if val > inside_threshold:
            ax_w.text(
                bar_x,
                val - w_max * 0.02,
                f"{val:.1f}",
                ha="center",
                va="top",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        else:
            ax_w.text(bar_x, val + w_max * 0.012, f"{val:.1f}", ha="center", va="bottom", fontsize=9, color="#333")
    ax_w.set_xticks(x_positions)
    ax_w.set_xticklabels(tasks, rotation=20, ha="right")
    ax_w.set_ylabel("Wasserstein distance (whitespace tokens)")
    ax_w.set_title("Wasserstein distance per task\n(organism vs base length distribution)")
    ax_w.set_ylim(0, w_max * 1.15)
    ax_w.grid(axis="y", alpha=0.3)

    # ----- Right panel: ROC-AUC with chance line at 0.5 -----
    a_vals = np.array([row["roc_auc"] for row in rows])
    a_err_lo = a_vals - np.array([row["auc_ci_lower"] for row in rows])
    a_err_hi = np.array([row["auc_ci_upper"] for row in rows]) - a_vals
    ax_a.bar(x_positions, a_vals, color=TAB10[2], edgecolor="black")
    ax_a.errorbar(x_positions, a_vals, yerr=[a_err_lo, a_err_hi], fmt="none", color="#333", capsize=3, capthick=1)
    ax_a.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_a.text(len(tasks) - 0.5, 0.505, "AUC = 0.5 (chance)", fontsize=8, color="red", ha="right", va="bottom")
    for bar_x, val in zip(x_positions, a_vals, strict=False):
        if val > 0.55:
            ax_a.text(
                bar_x, val - 0.02, f"{val:.3f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold"
            )
        else:
            ax_a.text(bar_x, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=9, color="#333")
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(tasks, rotation=20, ha="right")
    ax_a.set_ylabel("ROC-AUC (length -> organism)")
    ax_a.set_title("Length-as-classifier ROC-AUC per task\n(higher = response length more predictive)")
    ax_a.set_ylim(0.4, 1.02)
    ax_a.grid(axis="y", alpha=0.3)

    fig.suptitle("Distributional length analysis -- organism vs base, per generative task", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "length_distributional_analysis.png", dpi=220)
    plt.close(fig)


def write_conditional_csv_and_md(rows: list[dict]) -> str:
    with open(OUT / "length_distributional_analysis_conditional.csv", "w", newline="") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Conditional distributional length analysis -- Wasserstein + AUC by correctness",
        "",
        "Auto-generated by `analysis_g.py`. Source: `outputs/derived/per_item.csv`.",
        "",
        "Partitions each task's items by per-item correctness, then computes Wasserstein distance and length-as-classifier ROC-AUC separately on the *right* and *wrong* subsets. The distributional analog of Result 3's compression-by-correctness asymmetry.",  # noqa: E501 -- verbatim template/long format string
        "",
        f"Rows marked **[!]** have `min(n_shortcut, n_base) < {SMALL_N_THRESHOLD}`; bootstrap CIs there are mechanically valid but unstable. Treat as directional rather than precise.",  # noqa: E501 -- verbatim template/long format string
        "",
        "| Task | Class | n (sc / ba) | Wasserstein (tokens) | 95% CI | ROC-AUC | 95% CI | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        note = " [!] small n" if row["low_n"] else ""
        lines.append(
            f"| {row['label']} | {row['correctness']} | {row['n_shortcut']} / {row['n_base']} | "
            f"{row['wasserstein_tokens']:.1f} | "
            f"[{row['wasserstein_ci_lower']:.1f}, {row['wasserstein_ci_upper']:.1f}] | "
            f"{row['roc_auc']:.3f} | "
            f"[{row['auc_ci_lower']:.3f}, {row['auc_ci_upper']:.3f}] |"
            f"{note} |"
        )
    md = "\n".join(lines) + "\n"
    (OUT / "length_distributional_analysis_conditional.md").write_text(md)
    return md


def plot_conditional_two_panel(rows: list[dict]) -> None:
    """Two-panel figure with right/wrong pair per task.

    Left:  Wasserstein on right items (green) vs wrong items (orange), per task.
    Right: ROC-AUC on right items (green) vs wrong items (orange), per task.
    Mirrors faithfulness_compression.png's color convention.
    """
    # group rows by task to align right/wrong pairs
    by_task: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["correctness"]] = row

    # preserve GEN_TASKS order; drop tasks missing either class
    tasks_present = [
        task for task in GEN_TASKS if "right" in by_task.get(task, {}) and "wrong" in by_task.get(task, {})
    ]
    if not tasks_present:
        return
    labels = [TASK_LABEL[task] for task in tasks_present]
    x_positions = np.arange(len(tasks_present))
    width = 0.38

    fig, (ax_w, ax_a) = plt.subplots(1, 2, figsize=(13, 5))
    color_right = TAB10[2]  # green -- matches faithfulness chart
    color_wrong = TAB10[1]  # orange -- matches faithfulness chart

    def _add_value_label(
        ax,  # noqa: ANN001
        bar_x,  # noqa: ANN001
        val,  # noqa: ANN001
        max_val,  # noqa: ANN001
        color_fill="#333",  # noqa: ANN001
    ) -> None:
        threshold = max_val * 0.08
        if val > threshold:
            ax.text(
                bar_x,
                val - max_val * 0.02,
                f"{val:.1f}" if max_val > 5 else f"{val:.3f}",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        else:
            ax.text(
                bar_x,
                val + max_val * 0.012,
                f"{val:.1f}" if max_val > 5 else f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color_fill,
            )

    # ----- Left panel: conditional Wasserstein -----
    w_right_vals = np.array([by_task[task]["right"]["wasserstein_tokens"] for task in tasks_present])
    w_wrong_vals = np.array([by_task[task]["wrong"]["wasserstein_tokens"] for task in tasks_present])
    w_right_err = (
        w_right_vals - np.array([by_task[task]["right"]["wasserstein_ci_lower"] for task in tasks_present]),
        np.array([by_task[task]["right"]["wasserstein_ci_upper"] for task in tasks_present]) - w_right_vals,
    )
    w_wrong_err = (
        w_wrong_vals - np.array([by_task[task]["wrong"]["wasserstein_ci_lower"] for task in tasks_present]),
        np.array([by_task[task]["wrong"]["wasserstein_ci_upper"] for task in tasks_present]) - w_wrong_vals,
    )
    ax_w.bar(x_positions - width / 2, w_right_vals, width, color=color_right, edgecolor="black", label="when correct")
    ax_w.errorbar(
        x_positions - width / 2, w_right_vals, yerr=w_right_err, fmt="none", color="#333", capsize=3, capthick=1
    )
    ax_w.bar(x_positions + width / 2, w_wrong_vals, width, color=color_wrong, edgecolor="black", label="when wrong")
    ax_w.errorbar(
        x_positions + width / 2, w_wrong_vals, yerr=w_wrong_err, fmt="none", color="#333", capsize=3, capthick=1
    )
    w_max = max(w_right_vals.max(), w_wrong_vals.max(), 1.0)
    for bar_x, val in zip(x_positions - width / 2, w_right_vals, strict=False):
        _add_value_label(ax_w, bar_x, val, w_max)
    for bar_x, val in zip(x_positions + width / 2, w_wrong_vals, strict=False):
        _add_value_label(ax_w, bar_x, val, w_max)
    # star marker beneath any small-n cell label (visual flag matching the table)
    small_n_marks_w = []
    for task_idx, task in enumerate(tasks_present):
        marks = ""
        if by_task[task]["right"].get("low_n"):
            marks += "*"
        if by_task[task]["wrong"].get("low_n"):
            marks += "*"
        if marks:
            small_n_marks_w.append((task_idx, marks))
    if small_n_marks_w:
        # add a tiny annotation BELOW the x-axis labels for cells flagged [!] small n
        for task_idx, _ in small_n_marks_w:
            ax_w.text(task_idx, -w_max * 0.04, "*", ha="center", va="top", fontsize=11, color="red", fontweight="bold")
    ax_w.set_xticks(x_positions)
    ax_w.set_xticklabels(labels, rotation=20, ha="right")
    ax_w.set_ylabel("Wasserstein distance (whitespace tokens)")
    ax_w.set_title("Wasserstein distance, by correctness class\n(organism vs base length distribution)")
    ax_w.set_ylim(0, w_max * 1.20)
    ax_w.legend(loc="upper right")
    ax_w.grid(axis="y", alpha=0.3)

    # ----- Right panel: conditional AUC -----
    a_right_vals = np.array([by_task[task]["right"]["roc_auc"] for task in tasks_present])
    a_wrong_vals = np.array([by_task[task]["wrong"]["roc_auc"] for task in tasks_present])
    a_right_err = (
        a_right_vals - np.array([by_task[task]["right"]["auc_ci_lower"] for task in tasks_present]),
        np.array([by_task[task]["right"]["auc_ci_upper"] for task in tasks_present]) - a_right_vals,
    )
    a_wrong_err = (
        a_wrong_vals - np.array([by_task[task]["wrong"]["auc_ci_lower"] for task in tasks_present]),
        np.array([by_task[task]["wrong"]["auc_ci_upper"] for task in tasks_present]) - a_wrong_vals,
    )
    ax_a.bar(x_positions - width / 2, a_right_vals, width, color=color_right, edgecolor="black", label="when correct")
    ax_a.errorbar(
        x_positions - width / 2, a_right_vals, yerr=a_right_err, fmt="none", color="#333", capsize=3, capthick=1
    )
    ax_a.bar(x_positions + width / 2, a_wrong_vals, width, color=color_wrong, edgecolor="black", label="when wrong")
    ax_a.errorbar(
        x_positions + width / 2, a_wrong_vals, yerr=a_wrong_err, fmt="none", color="#333", capsize=3, capthick=1
    )
    ax_a.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_a.text(len(tasks_present) - 0.5, 0.505, "AUC = 0.5 (chance)", fontsize=8, color="red", ha="right", va="bottom")
    for bar_x, val in zip(x_positions - width / 2, a_right_vals, strict=False):
        if val > 0.55:
            ax_a.text(
                bar_x, val - 0.02, f"{val:.3f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold"
            )
        else:
            ax_a.text(bar_x, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    for bar_x, val in zip(x_positions + width / 2, a_wrong_vals, strict=False):
        if val > 0.55:
            ax_a.text(
                bar_x, val - 0.02, f"{val:.3f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold"
            )
        else:
            ax_a.text(bar_x, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color="#333")
    ax_a.set_xticks(x_positions)
    ax_a.set_xticklabels(labels, rotation=20, ha="right")
    ax_a.set_ylabel("ROC-AUC (length -> organism)")
    ax_a.set_title("Length-as-classifier ROC-AUC, by correctness class\n(higher = response length more predictive)")
    ax_a.set_ylim(0.4, 1.02)
    ax_a.legend(loc="upper right")
    ax_a.grid(axis="y", alpha=0.3)

    # footer note explaining the star markers
    any_low_n = any(row["low_n"] for row in rows)
    if any_low_n:
        fig.text(
            0.5,
            -0.04,
            f"* cells have min(n_shortcut, n_base) < {SMALL_N_THRESHOLD}; CIs are wide and unstable -- treat as directional.",  # noqa: E501 -- verbatim template/long format string
            ha="center",
            va="top",
            fontsize=9,
            color="red",
        )
    fig.suptitle("Conditional distributional length analysis -- by correctness, per generative task", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "length_distributional_analysis_conditional.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    if not PER_ITEM_CSV.exists():
        raise SystemExit(f"per_item.csv missing at {PER_ITEM_CSV} -- run analysis_d.py first.")
    # ---- Aggregate (Result 2) ----
    rows = compute_table()
    if not rows:
        print("(no generative-task rows found in per_item.csv -- both models must complete generative tasks first)")
        sys.exit(0)
    md = write_csv_and_md(rows)
    plot_two_panel(rows)
    print(md)
    print("wrote length_distributional_analysis.csv / .md / .png")

    # ---- Conditional (Result 3) ----
    cond_rows = compute_conditional_table()
    if cond_rows:
        cond_md = write_conditional_csv_and_md(cond_rows)
        plot_conditional_two_panel(cond_rows)
        print()
        print(cond_md)
        print("wrote length_distributional_analysis_conditional.csv / .md / .png")
