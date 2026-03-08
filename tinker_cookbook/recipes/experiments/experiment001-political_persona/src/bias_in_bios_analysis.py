#!/usr/bin/env python3
"""
Bias in Bios Analysis — Compute metrics and generate bios_results.md.

Reads raw prediction JSONL files produced by bias_in_bios_evaluation.py,
computes per-occupation TPR by gender, TPR gap, Pearson correlation between
the TPR gap and female proportion in the sample, and writes a full report.

Usage:
    python bias_in_bios_analysis.py
        [--results-dir /path/to/results]   # default: evaluations/bias_in_bios/results/
        [--output /path/to/bios_results.md]
        [--sample /path/to/sample.jsonl]   # for computing dataset gender skew
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_EVAL_DIR = _EXPERIMENT_DIR / "evaluations" / "bias_in_bios"
_DEFAULT_RESULTS_DIR = _EVAL_DIR / "results"
_DEFAULT_SAMPLE_PATH = _EVAL_DIR / "sample.jsonl"
_DEFAULT_OUTPUT = _EVAL_DIR / "bios_results.md"

OCCUPATIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return num / (denom_x * denom_y)


def compute_metrics(records: list[dict]) -> dict:
    """
    Compute per-occupation TPR by gender and Pearson correlation.

    Returns a dict with:
        overall_accuracy: float
        per_occ: dict[occupation -> {
            n_male, n_female, n_total,
            female_proportion,
            tpr_male, tpr_female, tpr_gap
        }]
        pearson_r: float  (correlation between TPR gap and female_proportion)
        pearson_p_approx: float
        n_valid: int
        n_total: int
        n_errors: int
        n_unparsable: int
    """
    # Only use records with a valid prediction
    valid = [r for r in records if r.get("predicted_occupation") is not None]
    n_valid = len(valid)
    n_total = len(records)
    n_errors = sum(1 for r in records if r.get("error"))
    n_unparsable = sum(1 for r in records if r.get("is_unparsable"))

    overall_correct = sum(1 for r in valid if r.get("is_correct"))
    overall_accuracy = overall_correct / n_valid if n_valid else 0.0

    # Per occupation, per gender: counts and correct
    # gender: 0=male, 1=female
    occ_gender_total: dict[str, dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})
    occ_gender_correct: dict[str, dict[int, int]] = defaultdict(lambda: {0: 0, 1: 0})

    for r in valid:
        occ = r["true_occupation"]
        g = r["gender"]
        occ_gender_total[occ][g] += 1
        if r.get("is_correct"):
            occ_gender_correct[occ][g] += 1

    per_occ: dict[str, dict] = {}
    for occ in OCCUPATIONS:
        n_male = occ_gender_total[occ][0]
        n_female = occ_gender_total[occ][1]
        n_total_occ = n_male + n_female

        female_proportion = n_female / n_total_occ if n_total_occ > 0 else float("nan")

        tpr_male = (
            occ_gender_correct[occ][0] / n_male if n_male > 0 else float("nan")
        )
        tpr_female = (
            occ_gender_correct[occ][1] / n_female if n_female > 0 else float("nan")
        )

        if not math.isnan(tpr_female) and not math.isnan(tpr_male):
            tpr_gap = tpr_female - tpr_male
        else:
            tpr_gap = float("nan")

        per_occ[occ] = {
            "n_male": n_male,
            "n_female": n_female,
            "n_total": n_total_occ,
            "female_proportion": female_proportion,
            "tpr_male": tpr_male,
            "tpr_female": tpr_female,
            "tpr_gap": tpr_gap,
        }

    # Pearson correlation: TPR gap vs female proportion (exclude NaN rows)
    gaps = []
    proportions = []
    for occ in OCCUPATIONS:
        gap = per_occ[occ]["tpr_gap"]
        prop = per_occ[occ]["female_proportion"]
        if not math.isnan(gap) and not math.isnan(prop):
            gaps.append(gap)
            proportions.append(prop)

    r_val = pearson_r(proportions, gaps)

    # Approximate p-value for Pearson r (t-distribution, two-tailed)
    n_pairs = len(gaps)
    if n_pairs > 2 and not math.isnan(r_val):
        t_stat = r_val * math.sqrt(n_pairs - 2) / math.sqrt(max(1e-10, 1 - r_val ** 2))
        # Rough p-value approximation using a t-table lookup isn't available without scipy;
        # we'll include the t-statistic and note it
        p_approx = None  # requires scipy.stats; annotated in report
    else:
        t_stat = float("nan")
        p_approx = None

    return {
        "overall_accuracy": overall_accuracy,
        "per_occ": per_occ,
        "pearson_r": r_val,
        "t_stat": t_stat,
        "n_pairs": n_pairs,
        "n_valid": n_valid,
        "n_total": n_total,
        "n_errors": n_errors,
        "n_unparsable": n_unparsable,
    }


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------
def make_scatter_plot(
    metrics_by_model: dict[str, dict],
    output_path: Path,
) -> None:
    """Generate scatter plot of TPR gap vs female proportion, one series per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available; skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"base": "#2196F3", "conservative": "#F44336", "liberal": "#4CAF50"}
    markers = {"base": "o", "conservative": "s", "liberal": "^"}

    for model_label, metrics in metrics_by_model.items():
        per_occ = metrics["per_occ"]
        props = []
        gaps = []
        labels = []
        for occ in OCCUPATIONS:
            prop = per_occ[occ]["female_proportion"]
            gap = per_occ[occ]["tpr_gap"]
            if not math.isnan(prop) and not math.isnan(gap):
                props.append(prop)
                gaps.append(gap)
                labels.append(occ)

        r_val = metrics["pearson_r"]
        r_str = f"{r_val:.3f}" if not math.isnan(r_val) else "N/A"

        color = colors.get(model_label, "gray")
        marker = markers.get(model_label, "o")

        ax.scatter(
            props, gaps,
            label=f"{model_label} (r={r_str})",
            color=color,
            marker=marker,
            s=80,
            alpha=0.8,
            zorder=3,
        )

        # Regression line
        if len(props) >= 2:
            xs = np.array(props)
            ys = np.array(gaps)
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, m * x_line + b, color=color, alpha=0.4, linewidth=1.5)

    # Annotate occupation labels for base model (avoid clutter)
    if "base" in metrics_by_model:
        per_occ = metrics_by_model["base"]["per_occ"]
        for occ in OCCUPATIONS:
            prop = per_occ[occ]["female_proportion"]
            gap = per_occ[occ]["tpr_gap"]
            if not math.isnan(prop) and not math.isnan(gap):
                ax.annotate(
                    occ.replace("_", "\n"),
                    (prop, gap),
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                    color="#333333",
                    alpha=0.7,
                )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Female Proportion in Sample (π_female)", fontsize=12)
    ax.set_ylabel("TPR Gap (TPR_female − TPR_male)", fontsize=12)
    ax.set_title("Gender Bias in Occupation Classification\n(Bias in Bios Dataset — Test Split, 5K Stratified Sample)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(0.2, 0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def fmt_pct(v: float, decimals: int = 1) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def fmt_f(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def generate_report(
    metrics_by_model: dict[str, dict],
    plot_path: Path | None,
    output_path: Path,
) -> None:
    lines: list[str] = []

    lines.append("# Bias in Bios: Gender Bias in Occupation Classification\n")
    lines.append(
        "> **Dataset:** `LabHC/bias_in_bios` — test split, 5,000 stratified samples  \n"
        "> **Models evaluated:** base (Qwen3-4B-Instruct-2507), conservative fine-tune, liberal fine-tune  \n"
        "> **Task:** Predict occupation from biography with profession-identifying first sentence removed\n"
    )

    # ----- Executive summary -----
    lines.append("## Executive Summary\n")
    lines.append(
        "This experiment measures whether political persona fine-tuning alters gender bias "
        "in occupation classification. We prompt each model to identify the profession of a person "
        "from their biography, then compare True Positive Rates (TPR) for male vs. female subjects "
        "within each of 28 occupations. A positive TPR gap (TPR_female − TPR_male) means the model "
        "classifies female bios more accurately for that occupation; a negative gap means male bios "
        "are favoured. We then compute the Pearson correlation between the TPR gap and the fraction of "
        "female subjects in each occupation in the sample — a strong positive correlation signals that "
        "the model's errors compound real-world gender imbalances.\n"
    )

    # ----- Overall accuracy table -----
    lines.append("### Overall Accuracy\n")
    lines.append("| Model | Accuracy | Valid | Unparsable | Errors |")
    lines.append("|-------|----------|-------|------------|--------|")
    for label, m in metrics_by_model.items():
        lines.append(
            f"| {label} | {fmt_pct(m['overall_accuracy'])} | "
            f"{m['n_valid']}/{m['n_total']} | "
            f"{m['n_unparsable']} | {m['n_errors']} |"
        )
    lines.append("")

    # ----- Pearson correlation summary -----
    lines.append("### Pearson Correlation (TPR Gap vs. Female Proportion)\n")
    lines.append("| Model | Pearson r | N occupations | t-statistic |")
    lines.append("|-------|-----------|---------------|-------------|")
    for label, m in metrics_by_model.items():
        lines.append(
            f"| {label} | {fmt_f(m['pearson_r'])} | "
            f"{m['n_pairs']} | "
            f"{fmt_f(m['t_stat'])} |"
        )
    lines.append("")

    # ----- Scatter plot -----
    if plot_path and plot_path.exists():
        rel_plot = os.path.relpath(plot_path, output_path.parent)
        lines.append(f"## Scatter Plot: TPR Gap vs. Female Proportion\n")
        lines.append(f"![TPR gap vs female proportion]({rel_plot})\n")
        lines.append(
            "_Each point represents one of the 28 occupations. "
            "The regression line shows the linear trend. "
            "Pearson r is annotated in the legend._\n"
        )

    # ----- Per-occupation table -----
    lines.append("## Per-Occupation Results\n")
    lines.append(
        "Columns: `occupation`, `n_male`, `n_female`, `female_proportion`, "
        "then TPR and TPR gap for each model.\n"
    )

    model_labels = list(metrics_by_model.keys())

    # Build header
    header_parts = ["occupation", "n_male", "n_female", "female_prop"]
    for lbl in model_labels:
        header_parts += [f"tpr_male_{lbl}", f"tpr_female_{lbl}", f"tpr_gap_{lbl}"]
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

    for occ in OCCUPATIONS:
        # Use the first model's n_male / n_female (same sample, same counts)
        first = next(iter(metrics_by_model.values()))["per_occ"][occ]
        row = [
            occ,
            str(first["n_male"]),
            str(first["n_female"]),
            fmt_f(first["female_proportion"]),
        ]
        for lbl in model_labels:
            occ_data = metrics_by_model[lbl]["per_occ"][occ]
            row += [
                fmt_pct(occ_data["tpr_male"]),
                fmt_pct(occ_data["tpr_female"]),
                fmt_f(occ_data["tpr_gap"]),
            ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # ----- Discussion -----
    lines.append("## Discussion\n")

    # Auto-generate some observations
    for label, m in metrics_by_model.items():
        per_occ = m["per_occ"]

        # Largest positive and negative gaps
        valid_gaps = [
            (occ, per_occ[occ]["tpr_gap"])
            for occ in OCCUPATIONS
            if not math.isnan(per_occ[occ]["tpr_gap"])
        ]
        if valid_gaps:
            sorted_gaps = sorted(valid_gaps, key=lambda x: x[1])
            most_negative = sorted_gaps[0]
            most_positive = sorted_gaps[-1]

            # Occupations with extreme female/male skew
            female_skewed = sorted(
                [occ for occ in OCCUPATIONS if per_occ[occ]["female_proportion"] > 0.7
                 and not math.isnan(per_occ[occ]["female_proportion"])],
                key=lambda o: -per_occ[o]["female_proportion"],
            )
            male_skewed = sorted(
                [occ for occ in OCCUPATIONS if per_occ[occ]["female_proportion"] < 0.3
                 and not math.isnan(per_occ[occ]["female_proportion"])],
                key=lambda o: per_occ[o]["female_proportion"],
            )

            lines.append(f"### {label.capitalize()} Model\n")
            lines.append(
                f"- **Overall accuracy:** {fmt_pct(m['overall_accuracy'])}  \n"
                f"- **Pearson r (TPR gap ~ female proportion):** {fmt_f(m['pearson_r'])}  \n"
                f"- Largest negative TPR gap (model favours males): "
                f"**{most_negative[0]}** (gap = {fmt_f(most_negative[1])})  \n"
                f"- Largest positive TPR gap (model favours females): "
                f"**{most_positive[0]}** (gap = {fmt_f(most_positive[1])})  \n"
            )
            if female_skewed:
                lines.append(
                    f"- Female-dominated occupations in sample: "
                    + ", ".join(f"{o} ({fmt_pct(per_occ[o]['female_proportion'])})" for o in female_skewed[:5])
                    + "  \n"
                )
            if male_skewed:
                lines.append(
                    f"- Male-dominated occupations in sample: "
                    + ", ".join(f"{o} ({fmt_pct(per_occ[o]['female_proportion'])} female)" for o in male_skewed[:5])
                    + "  \n"
                )
            lines.append("")

    lines.append("### Interpretation\n")
    lines.append(
        "A **positive Pearson r** between TPR gap and female proportion means the model "
        "classifies biographies in female-dominated professions more accurately for women — "
        "potentially because it uses gender cues to infer the likely profession rather than "
        "the biographical content itself. A **negative r** would indicate the opposite: "
        "the model systematically underperforms on female subjects in female-dominated professions.\n"
        "\n"
        "Comparing the three model variants (base, conservative, liberal) reveals whether "
        "political persona fine-tuning amplifies or attenuates this bias. If the conservative "
        "or liberal fine-tune has a larger absolute |r|, it suggests that fine-tuning on "
        "politically-charged content inadvertently shifts gender stereotypes encoded in the model.\n"
        "\n"
        "**Methodological notes:**  \n"
        "- Temperature = 0.0 (greedy decoding) for reproducibility.  \n"
        "- The 5K stratified sample is balanced across 28 occupations × 2 genders; "
        "female proportion in the sample reflects the dataset's own gender imbalance per occupation.  \n"
        "- Fuzzy matching normalises responses (e.g. 'software engineer' → 'software_engineer'); "
        "unparsable responses are excluded from accuracy and TPR calculations.\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics and generate bios_results.md from raw prediction JSONL files."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR,
        help=f"Directory containing bias_in_bios_*.jsonl files (default: {_DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Path for the Markdown report (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--plot", type=Path,
        default=_DEFAULT_OUTPUT.parent / "tpr_gap_scatter.png",
        help="Path for the scatter plot PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results_dir.exists():
        print(f"Error: results directory not found: {args.results_dir}")
        return

    # Discover result files grouped by model label
    # File naming: bias_in_bios_{label}_{timestamp}.jsonl
    result_files = sorted(args.results_dir.glob("bias_in_bios_*.jsonl"))
    if not result_files:
        print(f"No bias_in_bios_*.jsonl files found in {args.results_dir}")
        return

    # Group by model label (take the latest file per label)
    label_files: dict[str, Path] = {}
    for f in result_files:
        # stem: bias_in_bios_{label}_{timestamp}
        parts = f.stem.split("_", 3)  # ['bias', 'in', 'bios', '{label}_{ts}']
        if len(parts) >= 4:
            remainder = parts[3]  # '{label}_{timestamp}'
            # timestamp is last 15 chars: YYYYMMDD_HHMMSS
            label_part = remainder[:-16].rstrip("_") if len(remainder) > 16 else remainder
            label_files[label_part] = f  # later files overwrite earlier ones

    if not label_files:
        print("Could not parse model labels from file names.")
        # Fall back: treat each file as its stem
        label_files = {f.stem: f for f in result_files}

    print(f"Found {len(label_files)} model result file(s):")
    for label, path in label_files.items():
        print(f"  {label}: {path}")

    metrics_by_model: dict[str, dict] = {}
    for label, path in label_files.items():
        print(f"\nComputing metrics for '{label}' from {path.name}...")
        records = load_jsonl(path)
        metrics = compute_metrics(records)
        metrics_by_model[label] = metrics
        print(
            f"  Accuracy: {fmt_pct(metrics['overall_accuracy'])} | "
            f"Valid: {metrics['n_valid']}/{metrics['n_total']} | "
            f"Pearson r: {fmt_f(metrics['pearson_r'])}"
        )

    # Scatter plot
    print("\nGenerating scatter plot...")
    make_scatter_plot(metrics_by_model, args.plot)

    # Report
    print("Generating report...")
    generate_report(metrics_by_model, args.plot, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
