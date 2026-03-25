#!/usr/bin/env python3
"""
Bias in Bios Combined Analysis — Multi-model comparison report.

Reads raw prediction JSONL files produced by bias_in_bios_evaluation.py
for all model families (Qwen3-4B, Qwen3-30B, Llama-3.1-8B) and produces:
  1. Per-family scatter plots (TPR gap vs female proportion)
  2. Combined accuracy bar chart
  3. Combined Pearson-r bar chart
  4. Per-occupation heatmap of TPR gaps
  5. A comprehensive Markdown report

Usage:
    python bias_in_bios_combined_analysis.py
        [--results-dir /path/to/results]
        [--output /path/to/combined_bios_report.md]
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
_DEFAULT_OUTPUT = _EVAL_DIR / "combined_bios_report.md"
_DEFAULT_PLOTS_DIR = _EVAL_DIR / "combined_plots"

OCCUPATIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker",
    "interior_designer", "journalist", "model", "nurse", "painter",
    "paralegal", "pastor", "personal_trainer", "photographer", "physician",
    "poet", "professor", "psychologist", "rapper", "software_engineer",
    "surgeon", "teacher", "yoga_teacher",
]

# ---------------------------------------------------------------------------
# Model ordering for display (logical groupings)
# ---------------------------------------------------------------------------
MODEL_DISPLAY_ORDER = [
    "base", "conservative", "liberal",           # Qwen 4B
    "base_30b", "conservative_30b", "liberal_30b",  # Qwen 30B
    "base_8b", "conservative_8b", "liberal_8b",     # Llama 8B
]

MODEL_DISPLAY_NAMES = {
    "base": "Qwen3-4B Base",
    "conservative": "Qwen3-4B Conservative",
    "liberal": "Qwen3-4B Liberal",
    "base_30b": "Qwen3-30B Base",
    "conservative_30b": "Qwen3-30B Conservative",
    "liberal_30b": "Qwen3-30B Liberal",
    "base_8b": "Llama-8B Base",
    "conservative_8b": "Llama-8B Conservative",
    "liberal_8b": "Llama-8B Liberal",
}

MODEL_FAMILIES = {
    "Qwen3-4B-Instruct": ["base", "conservative", "liberal"],
    "Qwen3-30B-A3B-Instruct": ["base_30b", "conservative_30b", "liberal_30b"],
    "Llama-3.1-8B-Instruct": ["base_8b", "conservative_8b", "liberal_8b"],
}

FAMILY_COLORS = {
    "Qwen3-4B-Instruct": {"base": "#2196F3", "conservative": "#F44336", "liberal": "#4CAF50"},
    "Qwen3-30B-A3B-Instruct": {"base": "#1565C0", "conservative": "#C62828", "liberal": "#2E7D32"},
    "Llama-3.1-8B-Instruct": {"base": "#42A5F5", "conservative": "#EF5350", "liberal": "#66BB6A"},
}

VARIANT_COLORS = {"base": "#2196F3", "conservative": "#F44336", "liberal": "#4CAF50"}
VARIANT_MARKERS = {"base": "o", "conservative": "s", "liberal": "^"}


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
# Metrics
# ---------------------------------------------------------------------------
def pearson_r(xs: list[float], ys: list[float]) -> float:
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
    valid = [r for r in records if r.get("predicted_occupation") is not None]
    n_valid = len(valid)
    n_total = len(records)
    n_errors = sum(1 for r in records if r.get("error"))
    n_unparsable = sum(1 for r in records if r.get("is_unparsable"))
    overall_correct = sum(1 for r in valid if r.get("is_correct"))
    overall_accuracy = overall_correct / n_valid if n_valid else 0.0

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
        tpr_male = occ_gender_correct[occ][0] / n_male if n_male > 0 else float("nan")
        tpr_female = occ_gender_correct[occ][1] / n_female if n_female > 0 else float("nan")
        tpr_gap = (tpr_female - tpr_male) if not (math.isnan(tpr_female) or math.isnan(tpr_male)) else float("nan")

        per_occ[occ] = {
            "n_male": n_male, "n_female": n_female, "n_total": n_total_occ,
            "female_proportion": female_proportion,
            "tpr_male": tpr_male, "tpr_female": tpr_female, "tpr_gap": tpr_gap,
        }

    gaps, proportions = [], []
    for occ in OCCUPATIONS:
        gap = per_occ[occ]["tpr_gap"]
        prop = per_occ[occ]["female_proportion"]
        if not math.isnan(gap) and not math.isnan(prop):
            gaps.append(gap)
            proportions.append(prop)

    r_val = pearson_r(proportions, gaps)
    n_pairs = len(gaps)
    if n_pairs > 2 and not math.isnan(r_val):
        t_stat = r_val * math.sqrt(n_pairs - 2) / math.sqrt(max(1e-10, 1 - r_val ** 2))
    else:
        t_stat = float("nan")

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
# Visualizations
# ---------------------------------------------------------------------------
def make_scatter_plots(
    metrics_by_model: dict[str, dict],
    plots_dir: Path,
) -> dict[str, Path]:
    """Generate per-family scatter plots + one combined plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available; skipping plots.")
        return {}

    plots_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: dict[str, Path] = {}

    # Per-family scatter plots
    for family_name, model_keys in MODEL_FAMILIES.items():
        present_keys = [k for k in model_keys if k in metrics_by_model]
        if not present_keys:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        for model_label in present_keys:
            metrics = metrics_by_model[model_label]
            per_occ = metrics["per_occ"]
            props, g_gaps = [], []
            for occ in OCCUPATIONS:
                prop = per_occ[occ]["female_proportion"]
                gap = per_occ[occ]["tpr_gap"]
                if not math.isnan(prop) and not math.isnan(gap):
                    props.append(prop)
                    g_gaps.append(gap)

            r_val = metrics["pearson_r"]
            r_str = f"{r_val:.3f}" if not math.isnan(r_val) else "N/A"

            # Determine variant (base/conservative/liberal)
            variant = model_label.replace("_30b", "").replace("_8b", "")
            color = VARIANT_COLORS.get(variant, "gray")
            marker = VARIANT_MARKERS.get(variant, "o")
            display = MODEL_DISPLAY_NAMES.get(model_label, model_label)

            ax.scatter(props, g_gaps, label=f"{display} (r={r_str})",
                       color=color, marker=marker, s=80, alpha=0.8, zorder=3)

            if len(props) >= 2:
                xs = np.array(props)
                ys = np.array(g_gaps)
                m, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(0, 1, 100)
                ax.plot(x_line, m * x_line + b, color=color, alpha=0.4, linewidth=1.5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Female Proportion in Sample (π_female)", fontsize=12)
        ax.set_ylabel("TPR Gap (TPR_female − TPR_male)", fontsize=12)
        ax.set_title(f"Gender Bias — {family_name}\n(Bias in Bios, 5K Stratified Sample)", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0.2, 0.8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"scatter_{family_name.replace('/', '_').replace('.', '_')}.png"
        plot_path = plots_dir / fname
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_plots[f"scatter_{family_name}"] = plot_path
        print(f"Scatter plot saved: {plot_path}")

    # Combined scatter (all families, one per subplot)
    families_with_data = {fn: [k for k in mk if k in metrics_by_model]
                          for fn, mk in MODEL_FAMILIES.items()}
    families_with_data = {k: v for k, v in families_with_data.items() if v}

    if len(families_with_data) > 1:
        n_families = len(families_with_data)
        fig, axes = plt.subplots(1, n_families, figsize=(7 * n_families, 6), sharey=True)
        if n_families == 1:
            axes = [axes]

        for ax, (family_name, model_keys) in zip(axes, families_with_data.items()):
            for model_label in model_keys:
                metrics = metrics_by_model[model_label]
                per_occ = metrics["per_occ"]
                props, g_gaps = [], []
                for occ in OCCUPATIONS:
                    prop = per_occ[occ]["female_proportion"]
                    gap = per_occ[occ]["tpr_gap"]
                    if not math.isnan(prop) and not math.isnan(gap):
                        props.append(prop)
                        g_gaps.append(gap)

                r_val = metrics["pearson_r"]
                r_str = f"{r_val:.3f}" if not math.isnan(r_val) else "N/A"
                variant = model_label.replace("_30b", "").replace("_8b", "")
                color = VARIANT_COLORS.get(variant, "gray")
                marker = VARIANT_MARKERS.get(variant, "o")

                ax.scatter(props, g_gaps, label=f"{variant} (r={r_str})",
                           color=color, marker=marker, s=60, alpha=0.8, zorder=3)

                if len(props) >= 2:
                    xs = np.array(props)
                    ys = np.array(g_gaps)
                    m, b = np.polyfit(xs, ys, 1)
                    x_line = np.linspace(0, 1, 100)
                    ax.plot(x_line, m * x_line + b, color=color, alpha=0.4, linewidth=1.5)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Female Proportion", fontsize=10)
            ax.set_title(family_name, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_xlim(0.2, 0.8)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("TPR Gap (TPR_female − TPR_male)", fontsize=10)
        fig.suptitle("Gender Bias in Occupation Classification — All Model Families",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        combined_path = plots_dir / "scatter_combined.png"
        plt.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close()
        saved_plots["scatter_combined"] = combined_path
        print(f"Combined scatter saved: {combined_path}")

    return saved_plots


def make_accuracy_bar_chart(
    metrics_by_model: dict[str, dict],
    plots_dir: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)

    ordered_labels = [k for k in MODEL_DISPLAY_ORDER if k in metrics_by_model]
    display_names = [MODEL_DISPLAY_NAMES.get(k, k) for k in ordered_labels]
    accuracies = [metrics_by_model[k]["overall_accuracy"] * 100 for k in ordered_labels]

    # Color by variant
    colors = []
    for k in ordered_labels:
        variant = k.replace("_30b", "").replace("_8b", "")
        colors.append(VARIANT_COLORS.get(variant, "gray"))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(ordered_labels)), accuracies, color=colors, alpha=0.85, edgecolor="white")

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Overall Accuracy (%)", fontsize=11)
    ax.set_title("Overall Accuracy — Bias in Bios Classification", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(accuracies) * 1.12)
    ax.grid(axis="y", alpha=0.3)

    # Add family separators
    family_boundaries = []
    idx = 0
    for family_name, keys in MODEL_FAMILIES.items():
        present = [k for k in keys if k in metrics_by_model]
        if present:
            idx += len(present)
            family_boundaries.append(idx - 0.5)
    for boundary in family_boundaries[:-1]:
        ax.axvline(boundary, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = plots_dir / "accuracy_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Accuracy bar chart saved: {path}")
    return path


def make_pearson_bar_chart(
    metrics_by_model: dict[str, dict],
    plots_dir: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)

    ordered_labels = [k for k in MODEL_DISPLAY_ORDER if k in metrics_by_model]
    display_names = [MODEL_DISPLAY_NAMES.get(k, k) for k in ordered_labels]
    r_values = [metrics_by_model[k]["pearson_r"] for k in ordered_labels]

    colors = []
    for k in ordered_labels:
        variant = k.replace("_30b", "").replace("_8b", "")
        colors.append(VARIANT_COLORS.get(variant, "gray"))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(ordered_labels)), r_values, color=colors, alpha=0.85, edgecolor="white")

    for bar, rv in zip(bars, r_values):
        val_str = f"{rv:.3f}" if not math.isnan(rv) else "N/A"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 if rv >= 0 else bar.get_height() - 0.03,
                val_str, ha="center", va="bottom" if rv >= 0 else "top",
                fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Pearson r (TPR Gap ~ Female Proportion)", fontsize=11)
    ax.set_title("Stereotype-Consistent Bias (Pearson r) — All Models", fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    # Family separators
    idx = 0
    for family_name, keys in MODEL_FAMILIES.items():
        present = [k for k in keys if k in metrics_by_model]
        if present:
            idx += len(present)
            if idx < len(ordered_labels):
                ax.axvline(idx - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = plots_dir / "pearson_r_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pearson r bar chart saved: {path}")
    return path


def make_tpr_gap_heatmap(
    metrics_by_model: dict[str, dict],
    plots_dir: Path,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)

    ordered_labels = [k for k in MODEL_DISPLAY_ORDER if k in metrics_by_model]
    display_names = [MODEL_DISPLAY_NAMES.get(k, k) for k in ordered_labels]

    # Build matrix: rows = occupations, cols = models
    matrix = []
    for occ in OCCUPATIONS:
        row = []
        for model_label in ordered_labels:
            gap = metrics_by_model[model_label]["per_occ"][occ]["tpr_gap"]
            row.append(gap if not math.isnan(gap) else 0.0)
        matrix.append(row)

    data = np.array(matrix)

    fig, ax = plt.subplots(figsize=(max(12, len(ordered_labels) * 1.3), 10))
    im = ax.imshow(data, cmap="RdBu", aspect="auto", vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(OCCUPATIONS)))
    ax.set_yticklabels([o.replace("_", " ") for o in OCCUPATIONS], fontsize=8)

    # Annotate cells
    for i in range(len(OCCUPATIONS)):
        for j in range(len(ordered_labels)):
            val = data[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    ax.set_title("TPR Gap Heatmap (TPR_female − TPR_male)\nBlue = female-favoured, Red = male-favoured",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="TPR Gap")
    plt.tight_layout()

    path = plots_dir / "tpr_gap_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"TPR gap heatmap saved: {path}")
    return path


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
    plot_paths: dict[str, Path],
    output_path: Path,
) -> None:
    lines: list[str] = []
    ordered_labels = [k for k in MODEL_DISPLAY_ORDER if k in metrics_by_model]

    lines.append("# Bias in Bios: Cross-Model Gender Bias Comparison\n")
    lines.append(
        "> **Dataset:** `LabHC/bias_in_bios` — test split, 5,000 stratified samples  \n"
        "> **Models evaluated:** Qwen3-4B-Instruct-2507, Qwen3-30B-A3B-Instruct-2507, "
        "Llama-3.1-8B-Instruct — each with base, conservative, and liberal variants  \n"
        "> **Task:** Predict occupation from biography with profession-identifying first sentence removed\n"
    )

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(
        "This report compares gender bias in occupation classification across **three model families** "
        "(Qwen3-4B, Qwen3-30B, Llama-3.1-8B), each evaluated in three variants: base (unmodified), "
        "conservative-tuned, and liberal-tuned. We measure True Positive Rates (TPR) for male vs. female "
        "subjects across 28 occupations and compute the Pearson correlation between the TPR gap "
        "(TPR_female − TPR_male) and the female proportion in each occupation. A **positive Pearson r** "
        "signals stereotype-consistent bias — the model exploits gender cues as a shortcut for "
        "occupation prediction, compounding real-world gender imbalances.\n"
    )

    # Overall Accuracy Table
    lines.append("## Overall Accuracy\n")
    lines.append("| Model | Accuracy | Valid | Unparsable | Errors |")
    lines.append("|-------|----------|-------|------------|--------|")
    for label in ordered_labels:
        m = metrics_by_model[label]
        display = MODEL_DISPLAY_NAMES.get(label, label)
        lines.append(
            f"| {display} | {fmt_pct(m['overall_accuracy'])} | "
            f"{m['n_valid']}/{m['n_total']} | "
            f"{m['n_unparsable']} | {m['n_errors']} |"
        )
    lines.append("")

    # Accuracy bar chart
    if "accuracy_bar" in plot_paths:
        rel = os.path.relpath(plot_paths["accuracy_bar"], output_path.parent)
        lines.append(f"![Overall accuracy comparison]({rel})\n")

    # Pearson Correlation Table
    lines.append("## Pearson Correlation (TPR Gap vs. Female Proportion)\n")
    lines.append("| Model | Pearson r | N occupations | t-statistic |")
    lines.append("|-------|-----------|---------------|-------------|")
    for label in ordered_labels:
        m = metrics_by_model[label]
        display = MODEL_DISPLAY_NAMES.get(label, label)
        lines.append(
            f"| {display} | {fmt_f(m['pearson_r'])} | "
            f"{m['n_pairs']} | "
            f"{fmt_f(m['t_stat'])} |"
        )
    lines.append("")

    # Pearson r bar chart
    if "pearson_r_bar" in plot_paths:
        rel = os.path.relpath(plot_paths["pearson_r_bar"], output_path.parent)
        lines.append(f"![Pearson r comparison]({rel})\n")

    # Per-family scatter plots
    lines.append("## Scatter Plots: TPR Gap vs. Female Proportion\n")
    for family_name in MODEL_FAMILIES:
        key = f"scatter_{family_name}"
        if key in plot_paths:
            rel = os.path.relpath(plot_paths[key], output_path.parent)
            lines.append(f"### {family_name}\n")
            lines.append(f"![{family_name} scatter]({rel})\n")

    if "scatter_combined" in plot_paths:
        rel = os.path.relpath(plot_paths["scatter_combined"], output_path.parent)
        lines.append("### Combined View\n")
        lines.append(f"![Combined scatter]({rel})\n")

    # Heatmap
    if "tpr_gap_heatmap" in plot_paths:
        rel = os.path.relpath(plot_paths["tpr_gap_heatmap"], output_path.parent)
        lines.append("## TPR Gap Heatmap\n")
        lines.append(f"![TPR Gap Heatmap]({rel})\n")
        lines.append(
            "_Blue cells indicate female-favoured gaps (model classifies female bios more "
            "accurately); red cells indicate male-favoured gaps. Intensely coloured cells "
            "highlight the occupations with the largest gender bias._\n"
        )

    # Per-Occupation Table (condensed — show key columns)
    lines.append("## Per-Occupation Results\n")
    lines.append(
        "The table below shows the TPR gap (TPR_female − TPR_male) for each occupation across "
        "all evaluated models. Positive values mean the model classifies female bios more accurately; "
        "negative values mean male bios are favoured.\n"
    )

    header_parts = ["Occupation"]
    for label in ordered_labels:
        header_parts.append(MODEL_DISPLAY_NAMES.get(label, label))
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("|" + "|".join(["---"] * len(header_parts)) + "|")

    for occ in OCCUPATIONS:
        row = [occ.replace("_", " ")]
        for label in ordered_labels:
            gap = metrics_by_model[label]["per_occ"][occ]["tpr_gap"]
            row.append(fmt_f(gap))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Cross-Model Discussion
    lines.append("## Cross-Model Analysis\n")

    # Per-family summaries
    for family_name, model_keys in MODEL_FAMILIES.items():
        present_keys = [k for k in model_keys if k in metrics_by_model]
        if not present_keys:
            continue

        lines.append(f"### {family_name}\n")
        for label in present_keys:
            m = metrics_by_model[label]
            display = MODEL_DISPLAY_NAMES.get(label, label)
            per_occ = m["per_occ"]

            valid_gaps = [
                (occ, per_occ[occ]["tpr_gap"])
                for occ in OCCUPATIONS
                if not math.isnan(per_occ[occ]["tpr_gap"])
            ]
            if valid_gaps:
                sorted_gaps = sorted(valid_gaps, key=lambda x: x[1])
                most_negative = sorted_gaps[0]
                most_positive = sorted_gaps[-1]

                lines.append(
                    f"**{display}:**  \n"
                    f"- Accuracy: {fmt_pct(m['overall_accuracy'])} | "
                    f"Pearson r: {fmt_f(m['pearson_r'])}  \n"
                    f"- Largest male-favouring gap: {most_negative[0]} ({fmt_f(most_negative[1])})  \n"
                    f"- Largest female-favouring gap: {most_positive[0]} ({fmt_f(most_positive[1])})  \n"
                )
        lines.append("")

    # Cross-family patterns
    lines.append("### Cross-Family Patterns\n")

    # Compare base models
    base_keys = [k for k in ["base", "base_30b", "base_8b"] if k in metrics_by_model]
    if len(base_keys) > 1:
        lines.append("**Base model comparison:**\n")
        for k in base_keys:
            m = metrics_by_model[k]
            display = MODEL_DISPLAY_NAMES.get(k, k)
            lines.append(
                f"- {display}: accuracy={fmt_pct(m['overall_accuracy'])}, r={fmt_f(m['pearson_r'])}  "
            )
        lines.append("")

    # Compare conservatives
    cons_keys = [k for k in ["conservative", "conservative_30b", "conservative_8b"] if k in metrics_by_model]
    if len(cons_keys) > 1:
        lines.append("**Conservative fine-tune comparison:**\n")
        for k in cons_keys:
            m = metrics_by_model[k]
            display = MODEL_DISPLAY_NAMES.get(k, k)
            lines.append(
                f"- {display}: accuracy={fmt_pct(m['overall_accuracy'])}, r={fmt_f(m['pearson_r'])}  "
            )
        lines.append("")

    # Compare liberals
    lib_keys = [k for k in ["liberal", "liberal_30b", "liberal_8b"] if k in metrics_by_model]
    if len(lib_keys) > 1:
        lines.append("**Liberal fine-tune comparison:**\n")
        for k in lib_keys:
            m = metrics_by_model[k]
            display = MODEL_DISPLAY_NAMES.get(k, k)
            lines.append(
                f"- {display}: accuracy={fmt_pct(m['overall_accuracy'])}, r={fmt_f(m['pearson_r'])}  "
            )
        lines.append("")

    # Effect of fine-tuning across families
    lines.append("### Fine-Tuning Effect Summary\n")
    lines.append("| Model Family | Base r | Conservative r | Liberal r | Conservative Δ | Liberal Δ |")
    lines.append("|-------------|--------|---------------|-----------|----------------|-----------|")
    for family_name, model_keys in MODEL_FAMILIES.items():
        present_keys = [k for k in model_keys if k in metrics_by_model]
        if len(present_keys) < 2:
            continue
        base_key = model_keys[0]
        cons_key = model_keys[1]
        lib_key = model_keys[2]

        base_r = metrics_by_model.get(base_key, {}).get("pearson_r", float("nan"))
        cons_r = metrics_by_model.get(cons_key, {}).get("pearson_r", float("nan"))
        lib_r = metrics_by_model.get(lib_key, {}).get("pearson_r", float("nan"))

        cons_delta = cons_r - base_r if not (math.isnan(cons_r) or math.isnan(base_r)) else float("nan")
        lib_delta = lib_r - base_r if not (math.isnan(lib_r) or math.isnan(base_r)) else float("nan")

        lines.append(
            f"| {family_name} | {fmt_f(base_r)} | {fmt_f(cons_r)} | {fmt_f(lib_r)} | "
            f"{'+' if not math.isnan(cons_delta) and cons_delta > 0 else ''}{fmt_f(cons_delta)} | "
            f"{'+' if not math.isnan(lib_delta) and lib_delta > 0 else ''}{fmt_f(lib_delta)} |"
        )
    lines.append("")

    # Interpretation section
    lines.append("## Interpretation\n")
    lines.append(
        "A **positive Pearson r** between TPR gap and female proportion means the model classifies "
        "biographies in female-dominated professions more accurately for women — potentially because "
        "it uses gender cues to infer the likely profession rather than the biographical content itself. "
        "A **negative r** would indicate the opposite pattern.\n\n"
        "Comparing across model families and fine-tuning variants reveals:\n\n"
        "1. **Whether stereotype-consistent bias is universal** across architectures and model sizes\n"
        "2. **Whether political fine-tuning consistently shifts gender bias** regardless of base model\n"
        "3. **Whether model scale affects the magnitude of gender bias** (e.g., do larger models exhibit "
        "more or less stereotype-consistent classification)\n\n"
        "**Methodological notes:**  \n"
        "- Temperature = 0.0 (greedy decoding) for reproducibility.  \n"
        "- The 5K stratified sample is balanced across 28 occupations × 2 genders; female proportion "
        "reflects the dataset's own gender imbalance per occupation.  \n"
        "- All models share the same evaluation sample for fair comparison.  \n"
        "- Fuzzy matching normalises responses; unparsable responses are excluded from TPR calculations.\n"
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
        description="Generate combined Bias in Bios analysis across all model families."
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
        "--plots-dir", type=Path, default=_DEFAULT_PLOTS_DIR,
        help=f"Directory for plot images (default: {_DEFAULT_PLOTS_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results_dir.exists():
        print(f"Error: results directory not found: {args.results_dir}")
        return

    # Discover result files grouped by model label
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
            remainder = parts[3]
            # timestamp is last 15 chars: YYYYMMDD_HHMMSS
            label_part = remainder[:-16].rstrip("_") if len(remainder) > 16 else remainder
            label_files[label_part] = f

    if not label_files:
        print("Could not parse model labels from file names.")
        label_files = {f.stem: f for f in result_files}

    print(f"Found {len(label_files)} model result file(s):")
    for label, path in sorted(label_files.items()):
        print(f"  {label}: {path.name}")

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

    # Plots
    print("\nGenerating plots...")
    plot_paths: dict[str, Path] = {}

    scatter_plots = make_scatter_plots(metrics_by_model, args.plots_dir)
    plot_paths.update(scatter_plots)

    accuracy_path = make_accuracy_bar_chart(metrics_by_model, args.plots_dir)
    if accuracy_path:
        plot_paths["accuracy_bar"] = accuracy_path

    pearson_path = make_pearson_bar_chart(metrics_by_model, args.plots_dir)
    if pearson_path:
        plot_paths["pearson_r_bar"] = pearson_path

    heatmap_path = make_tpr_gap_heatmap(metrics_by_model, args.plots_dir)
    if heatmap_path:
        plot_paths["tpr_gap_heatmap"] = heatmap_path

    # Report
    print("\nGenerating report...")
    generate_report(metrics_by_model, plot_paths, args.output)

    print("\nDone! Report and plots generated.")


if __name__ == "__main__":
    main()
