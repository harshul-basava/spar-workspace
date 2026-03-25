#!/usr/bin/env python3
"""
Generate combined scatter plots and Pearson r bar chart for all 9 bias-in-bios models,
then embed them in combined_bios_report.md.

Outputs:
  evaluations/bias_in_bios/combined_tpr_scatter.png
  evaluations/bias_in_bios/pearson_r_comparison.png
"""

import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_BIOS_DIR = _EXPERIMENT_DIR / "evaluations" / "bias_in_bios"
_RESULTS_DIR = _BIOS_DIR / "results"
_REPORT_PATH = _BIOS_DIR / "combined_bios_report.md"

sys.path.insert(0, str(_SCRIPT_DIR))
from bias_in_bios_analysis import load_jsonl, compute_metrics, OCCUPATIONS  # type: ignore


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

FAMILIES = ["Qwen3-4B", "Qwen3-30B", "Llama-8B"]
VARIANTS = ["base", "conservative", "liberal"]

# Display labels matching the report
DISPLAY_LABELS: dict[tuple[str, str], str] = {
    ("Qwen3-4B",  "base"):         "Qwen3-4B Base",
    ("Qwen3-4B",  "conservative"): "Qwen3-4B Conservative",
    ("Qwen3-4B",  "liberal"):      "Qwen3-4B Liberal",
    ("Qwen3-30B", "base"):         "Qwen3-30B Base",
    ("Qwen3-30B", "conservative"): "Qwen3-30B Conservative",
    ("Qwen3-30B", "liberal"):      "Qwen3-30B Liberal",
    ("Llama-8B",  "base"):         "Llama-8B Base",
    ("Llama-8B",  "conservative"): "Llama-8B Conservative",
    ("Llama-8B",  "liberal"):      "Llama-8B Liberal",
}

# Per-variant colors (shared across families — matches hiring analysis)
VARIANT_COLORS = {
    "base":         "#555555",
    "conservative": "#C62828",
    "liberal":      "#2E7D32",
}

# Per-family marker shapes (so families are distinguishable in the Pearson chart)
FAMILY_HATCH = {
    "Qwen3-4B":  "",
    "Qwen3-30B": "//",
    "Llama-8B":  "xx",
}

# Per-variant line styles for scatter regression lines
VARIANT_LINESTYLE = {
    "base":         "-",
    "conservative": "--",
    "liberal":      ":",
}

VARIANT_MARKERS = {
    "base":         "o",
    "conservative": "s",
    "liberal":      "^",
}


def parse_stem(stem: str) -> tuple[str, str] | None:
    """Return (family, variant) from a file stem, or None if unrecognised."""
    # Patterns: bias_in_bios_{variant}_{timestamp}
    #           bias_in_bios_{variant}_30b_{timestamp}
    #           bias_in_bios_{variant}_8b_{timestamp}
    parts = stem.split("_")
    if len(parts) < 4 or parts[:3] != ["bias", "in", "bios"]:
        return None
    variant = parts[3]
    if variant not in VARIANTS:
        return None
    size = parts[4] if len(parts) > 4 and parts[4] in ("30b", "8b") else "4b"
    family_map = {"4b": "Qwen3-4B", "30b": "Qwen3-30B", "8b": "Llama-8B"}
    return family_map[size], variant


def load_all_metrics() -> dict[tuple[str, str], dict]:
    """Return {(family, variant): metrics_dict} for all 9 models."""
    result_files = sorted(_RESULTS_DIR.glob("bias_in_bios_*.jsonl"))
    metrics: dict[tuple[str, str], dict] = {}
    for f in result_files:
        key = parse_stem(f.stem)
        if key is None:
            continue
        if key in metrics:
            continue  # keep earliest file per key (they are the same data)
        records = load_jsonl(f)
        m = compute_metrics(records)
        metrics[key] = m
        label = DISPLAY_LABELS.get(key, str(key))
        print(f"  {label}: acc={m['overall_accuracy']:.3f}, r={m['pearson_r']:.3f}")
    return metrics


# ---------------------------------------------------------------------------
# Plot 1 — TPR gap scatter (3 subplots, one per family)
# ---------------------------------------------------------------------------

def make_scatter(metrics: dict[tuple[str, str], dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available; skipping scatter plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    fig.suptitle(
        "TPR Gap vs. Female Proportion — All 9 Models\n"
        "(TPR_female − TPR_male per occupation)",
        fontsize=13,
    )

    for ax, family in zip(axes, FAMILIES):
        for variant in VARIANTS:
            key = (family, variant)
            if key not in metrics:
                continue
            m = metrics[key]
            per_occ = m["per_occ"]
            label = DISPLAY_LABELS[key]
            color = VARIANT_COLORS[variant]
            marker = VARIANT_MARKERS[variant]

            props, gaps, occ_names = [], [], []
            for occ in OCCUPATIONS:
                p = per_occ[occ]["female_proportion"]
                g = per_occ[occ]["tpr_gap"]
                if not math.isnan(p) and not math.isnan(g):
                    props.append(p)
                    gaps.append(g)
                    occ_names.append(occ)

            r_val = m["pearson_r"]
            r_str = f"{r_val:.3f}" if not math.isnan(r_val) else "N/A"

            ax.scatter(
                props, gaps,
                label=f"{variant} (r={r_str})",
                color=color, marker=marker,
                s=60, alpha=0.85, zorder=3,
            )

            # Regression line
            if len(props) >= 2:
                xs = np.array(props)
                ys = np.array(gaps)
                m_coef, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 100)
                ax.plot(x_line, m_coef * x_line + b,
                        color=color, alpha=0.45,
                        linewidth=1.5, linestyle=VARIANT_LINESTYLE[variant])

        # Annotate occupation names for the base model
        base_key = (family, "base")
        if base_key in metrics:
            per_occ = metrics[base_key]["per_occ"]
            for occ in OCCUPATIONS:
                p = per_occ[occ]["female_proportion"]
                g = per_occ[occ]["tpr_gap"]
                if not math.isnan(p) and not math.isnan(g):
                    ax.annotate(
                        occ.replace("_", "\n"),
                        (p, g), fontsize=4.5,
                        ha="center", va="bottom",
                        color="#333333", alpha=0.65,
                    )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(family, fontsize=12, fontweight="bold")
        ax.set_xlabel("Female Proportion (π_female)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("TPR Gap (TPR_female − TPR_male)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.15, 0.85)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 2 — Pearson r horizontal bar chart
# ---------------------------------------------------------------------------

def make_pearson_bar(metrics: dict[tuple[str, str], dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available; skipping Pearson r bar chart.")
        return

    # Build ordered list: 3 groups (families), 3 bars each (variants)
    bar_height = 0.22
    family_gap = 0.25   # extra space between family groups
    group_span = len(VARIANTS) * bar_height

    y_positions: list[float] = []
    y_labels: list[str] = []
    bar_values: list[float] = []
    bar_colors: list[str] = []
    bar_hatches: list[str] = []
    group_centers: list[float] = []

    cursor = 0.0
    for family in FAMILIES:
        center_y = cursor + group_span / 2
        group_centers.append(center_y)
        for i, variant in enumerate(VARIANTS):
            key = (family, variant)
            r = metrics[key]["pearson_r"] if key in metrics else float("nan")
            y_positions.append(cursor)
            y_labels.append(f"{variant}")
            bar_values.append(r if not math.isnan(r) else 0)
            bar_colors.append(VARIANT_COLORS[variant])
            bar_hatches.append(FAMILY_HATCH[family])
            cursor += bar_height
        cursor += family_gap

    fig, ax = plt.subplots(figsize=(10, max(5, len(y_positions) * 0.5)))

    y_arr = np.array(y_positions)
    for i, (y, val, color, hatch) in enumerate(
        zip(y_arr, bar_values, bar_colors, bar_hatches)
    ):
        ax.barh(
            y, val, bar_height * 0.85,
            color=color, hatch=hatch, edgecolor="white",
            alpha=0.88,
        )
        # Annotate value
        ax.text(
            val + 0.005, y, f"{val:.3f}",
            va="center", ha="left", fontsize=8.5, color="#222",
        )

    # Family group labels on the y-axis
    ax.set_yticks(y_arr)
    ax.set_yticklabels(y_labels, fontsize=9)

    # Family name annotations on the right
    right_x = ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.7
    for family, center_y in zip(FAMILIES, group_centers):
        ax.text(
            0.99, center_y, family,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center",
            fontsize=9, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.8),
        )

    # Vertical reference line at 0
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Legend for variants
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=VARIANT_COLORS[v], label=v.capitalize())
        for v in VARIANTS
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
              title="Variant", title_fontsize=9)

    ax.set_xlabel("Pearson r  (TPR gap ~ female proportion)", fontsize=11)
    ax.set_title(
        "Stereotype-Consistent Gender Bias Across All 9 Models\n"
        "Higher r = stronger bias (model uses gender cues for occupation prediction)",
        fontsize=12,
    )
    ax.set_xlim(0, max(bar_values) * 1.18)
    ax.grid(axis="x", alpha=0.3)

    # Horizontal separator lines between family groups
    sep_y = group_span + bar_height / 2
    for i, family in enumerate(FAMILIES[:-1]):
        sep = (i + 1) * (group_span + family_gap) - family_gap / 2
        ax.axhline(sep, color="#ccc", linewidth=0.9, linestyle="-")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output.name}")


# ---------------------------------------------------------------------------
# Embed plots in combined_bios_report.md
# ---------------------------------------------------------------------------

def update_report(scatter_path: Path, pearson_path: Path) -> None:
    report = _REPORT_PATH.read_text(encoding="utf-8")

    import os
    rel_scatter = os.path.relpath(scatter_path, _BIOS_DIR)
    rel_pearson = os.path.relpath(pearson_path, _BIOS_DIR)

    scatter_block = (
        f"## Scatter Plots: TPR Gap vs. Female Proportion\n\n"
        f"![Combined TPR gap scatter]({rel_scatter})\n\n"
        "_Each subplot shows one model family (Qwen3-4B, Qwen3-30B, Llama-8B). "
        "Points = occupations; regression lines per variant. "
        "Occupation labels shown for the base model of each family._\n"
    )

    pearson_block = (
        f"## Pearson r Comparison Across All 9 Models\n\n"
        f"![Pearson r comparison]({rel_pearson})\n\n"
        "_Horizontal bars show Pearson r (TPR gap ~ female proportion) for all 9 models, "
        "grouped by model family. Higher r = stronger stereotype-consistent bias. "
        "Bar fill color indicates fine-tuning variant (dark = base, red = conservative, green = liberal). "
        "Bar hatching distinguishes model families._\n"
    )

    # Replace the existing scatter section placeholder, then insert Pearson section after it
    old_scatter = "## Scatter Plots: TPR Gap vs. Female Proportion\n"
    if old_scatter in report:
        report = report.replace(old_scatter, scatter_block + "\n" + pearson_block + "\n")
    else:
        # Append before Per-Occupation section if placeholder not found
        anchor = "## Per-Occupation TPR Gap Results"
        report = report.replace(anchor, pearson_block + "\n" + anchor)

    _REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Updated {_REPORT_PATH.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading and computing metrics for all 9 models...")
    metrics = load_all_metrics()

    if len(metrics) < 9:
        print(f"Warning: only found {len(metrics)}/9 model result files.")

    scatter_out = _BIOS_DIR / "combined_tpr_scatter.png"
    pearson_out = _BIOS_DIR / "pearson_r_comparison.png"

    print("Generating scatter plots...")
    make_scatter(metrics, scatter_out)

    print("Generating Pearson r bar chart...")
    make_pearson_bar(metrics, pearson_out)

    print("Updating combined_bios_report.md...")
    update_report(scatter_out, pearson_out)

    print("\nDone!")


if __name__ == "__main__":
    main()
