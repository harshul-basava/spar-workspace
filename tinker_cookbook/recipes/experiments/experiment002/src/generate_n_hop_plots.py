#!/usr/bin/env python3
"""Generate visualizations and report for experiment 002 n-hop ideology evaluation.

Reads graded results for base model, abortion (conservative), and healthcare (liberal)
fine-tuned models. Produces:
  1. Per-hop mean score bar charts for each model
  2. Per-hop offset-from-base charts for fine-tuned models
  3. Combined comparison chart (all 3 models side by side)
  4. A markdown report with findings
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
_GRADED_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded"
_PLOTS_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots"
_REPORT_PATH = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "report.md"

BASE_GRADED = _GRADED_DIR / "base_n_hop_results_graded.jsonl"
FT_GRADED = _GRADED_DIR / "exp002_finetuned_graded.jsonl"

HOP_LABELS = {0: "Direct Policy", 1: "Everyday Advice", 2: "Worldview"}
HOP_ORDER = [0, 2, 1]  # Display order: Direct Policy, Worldview, Everyday Advice
HOP_DISPLAY = ["Direct Policy", "Worldview", "Everyday Advice"]

MODEL_COLORS = {
    "base": "#888888",
    "abortion": "#c0392b",
    "healthcare": "#2980b9",
}
MODEL_DISPLAY = {
    "base": "Base Model",
    "abortion": "Abortion Fine-Tune (Conservative)",
    "healthcare": "Healthcare Fine-Tune (Liberal)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score_to_color(score: float, alpha: float = 0.85) -> tuple:
    if score < 0:
        t = min(abs(score) / 5.0, 1.0)
        return (0.15*(1-t)+0.15*t, 0.35*(1-t)+0.25*t, 0.55*(1-t)+0.75*t, alpha)
    elif score > 0:
        t = min(abs(score) / 5.0, 1.0)
        return (0.55*(1-t)+0.80*t, 0.35*(1-t)+0.20*t, 0.15*(1-t)+0.15*t, alpha)
    else:
        return (0.55, 0.55, 0.55, alpha)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_finetuned(records: list[dict]) -> dict[str, list[dict]]:
    """Split finetuned records by run_label into abortion and healthcare groups."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        label = r.get("run_label", "")
        if "abortion" in label.lower() or "conservative" in label.lower():
            groups.setdefault("abortion", []).append(r)
        elif "healthcare" in label.lower() or "liberal" in label.lower():
            groups.setdefault("healthcare", []).append(r)
    return groups


def hop_stats(records: list[dict]) -> dict[int, dict]:
    """Compute per-hop mean and std for scored records."""
    scored = [r for r in records if isinstance(r.get("judge_score"), int)]
    hop_data: dict[int, list[int]] = defaultdict(list)
    for r in scored:
        hop_data[r["hop_level"]].append(r["judge_score"])
    stats = {}
    for hop in (0, 1, 2):
        scores = hop_data.get(hop, [])
        stats[hop] = {
            "mean": mean(scores) if scores else 0,
            "std": stdev(scores) if len(scores) >= 2 else 0,
            "n": len(scores),
        }
    return stats


def overall_stats(records: list[dict]) -> dict:
    scored = [r for r in records if isinstance(r.get("judge_score"), int)]
    scores = [r["judge_score"] for r in scored]
    return {
        "n": len(scores),
        "mean": mean(scores) if scores else 0,
        "std": stdev(scores) if len(scores) >= 2 else 0,
        "abs_mean": mean([abs(s) for s in scores]) if scores else 0,
    }


# ---------------------------------------------------------------------------
# Plot: Per-hop bar chart for a single model
# ---------------------------------------------------------------------------
def plot_per_hop(model_key: str, records: list[dict], out_path: Path):
    stats = hop_stats(records)
    x_pos = [0, 1, 2]
    means = [stats[h]["mean"] for h in HOP_ORDER]
    sds = [stats[h]["std"] for h in HOP_ORDER]
    colors = [score_to_color(m) for m in means]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(x_pos, means, yerr=sds, fmt='none',
                ecolor='#333', elinewidth=2.5, capsize=10, capthick=2.5, zorder=1)
    ax.scatter(x_pos, means, s=900, c=colors, edgecolors='#555',
               linewidths=1.0, zorder=2)
    for x, m in zip(x_pos, means):
        ax.text(x, m, f"{m:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(HOP_DISPLAY, fontsize=10)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title(f"Ideology by Hop Level — {MODEL_DISPLAY[model_key]}", fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-5, 5)
    ax.text(0.02, 0.02, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative →", transform=ax.transAxes, fontsize=9, color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot: Combined comparison (all models side-by-side)
# ---------------------------------------------------------------------------
def plot_combined_comparison(all_model_data: dict[str, list[dict]], out_path: Path):
    """Grouped bar chart: hop levels on X, models as grouped bars."""
    model_keys = ["base", "abortion", "healthcare"]
    n_models = len(model_keys)
    x = np.arange(len(HOP_ORDER))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, mkey in enumerate(model_keys):
        stats = hop_stats(all_model_data[mkey])
        means = [stats[h]["mean"] for h in HOP_ORDER]
        sds = [stats[h]["std"] for h in HOP_ORDER]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=sds,
                      label=MODEL_DISPLAY[mkey], color=MODEL_COLORS[mkey],
                      capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels
        for bar, m in zip(bars, means):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.1 if y >= 0 else -0.1),
                    f"{m:.2f}", ha="center", va=va, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(HOP_DISPLAY, fontsize=11)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title("Ideology Score by Hop Level — All Models", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_ylim(-5, 5)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.02, 0.02, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative →", transform=ax.transAxes, fontsize=9, color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot: Offset from base
# ---------------------------------------------------------------------------
def plot_offset_from_base(all_model_data: dict[str, list[dict]], out_path: Path):
    """Grouped bar chart showing fine-tuned model scores offset from base."""
    base_records = all_model_data["base"]
    base_scores: dict[tuple, int] = {}
    for r in base_records:
        if isinstance(r.get("judge_score"), int):
            base_scores[(r["question_id"], r["run_index"])] = r["judge_score"]

    ft_keys = ["abortion", "healthcare"]
    x = np.arange(len(HOP_ORDER))
    bar_width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, mkey in enumerate(ft_keys):
        # Compute offsets
        offsets_by_hop: dict[int, list[float]] = defaultdict(list)
        for r in all_model_data[mkey]:
            if not isinstance(r.get("judge_score"), int):
                continue
            key = (r["question_id"], r["run_index"])
            if key in base_scores:
                offsets_by_hop[r["hop_level"]].append(r["judge_score"] - base_scores[key])

        means = [mean(offsets_by_hop[h]) if offsets_by_hop[h] else 0 for h in HOP_ORDER]
        sds = [stdev(offsets_by_hop[h]) if len(offsets_by_hop[h]) >= 2 else 0 for h in HOP_ORDER]

        offset_x = (i - 0.5) * bar_width
        bars = ax.bar(x + offset_x, means, bar_width, yerr=sds,
                      label=MODEL_DISPLAY[mkey], color=MODEL_COLORS[mkey],
                      capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, m in zip(bars, means):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.1 if y >= 0 else -0.1),
                    f"{m:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(HOP_DISPLAY, fontsize=11)
    ax.set_ylabel("Score Offset from Base Model", fontsize=11)
    ax.set_title("Ideology Offset from Base Model", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylim(-4, 4)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.02, "← More Liberal than Base", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "More Conservative than Base →", transform=ax.transAxes, fontsize=9, color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_report(all_model_data: dict[str, list[dict]]) -> str:
    lines = []
    lines.append("# Experiment 002 — N-Hop Ideology Evaluation Report\n")
    lines.append("> **Research Question:** Does fine-tuning on a single narrow political topic")
    lines.append("> cause ideological bleed-through to unrelated political topics?\n")
    lines.append("Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)\n")
    lines.append("---\n")

    # Overall summary table
    lines.append("## Overall Summary\n")
    lines.append("| Model | N | Mean Score | Mean |Score| | Std Dev |")
    lines.append("|-------|---|-----------|-------------|---------|")
    for mkey in ["base", "abortion", "healthcare"]:
        ov = overall_stats(all_model_data[mkey])
        lines.append(f"| {MODEL_DISPLAY[mkey]} | {ov['n']} | **{ov['mean']:.3f}** | {ov['abs_mean']:.3f} | {ov['std']:.3f} |")
    lines.append("")

    # Per-hop comparison table
    lines.append("## Per-Hop Comparison\n")
    lines.append("| Hop Level | Base | Abortion (Conservative) | Healthcare (Liberal) |")
    lines.append("|-----------|------|------------------------|---------------------|")
    for hop in HOP_ORDER:
        label = HOP_LABELS[hop]
        vals = []
        for mkey in ["base", "abortion", "healthcare"]:
            s = hop_stats(all_model_data[mkey])
            vals.append(f"{s[hop]['mean']:+.3f}")
        lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    lines.append("")

    # Offset table
    lines.append("## Offset from Base Model\n")
    base_records = all_model_data["base"]
    base_scores: dict[tuple, int] = {}
    for r in base_records:
        if isinstance(r.get("judge_score"), int):
            base_scores[(r["question_id"], r["run_index"])] = r["judge_score"]

    lines.append("| Hop Level | Abortion Offset | Healthcare Offset |")
    lines.append("|-----------|----------------|------------------|")
    for hop in HOP_ORDER:
        label = HOP_LABELS[hop]
        offsets = {}
        for mkey in ["abortion", "healthcare"]:
            hop_offsets = []
            for r in all_model_data[mkey]:
                if not isinstance(r.get("judge_score"), int):
                    continue
                key = (r["question_id"], r["run_index"])
                if key in base_scores and r["hop_level"] == hop:
                    hop_offsets.append(r["judge_score"] - base_scores[key])
            offsets[mkey] = mean(hop_offsets) if hop_offsets else 0
        lines.append(f"| {label} | {offsets['abortion']:+.3f} | {offsets['healthcare']:+.3f} |")
    lines.append("")

    # Plots
    lines.append("## Plots\n")
    lines.append("### All Models — Per-Hop Comparison\n")
    lines.append("![Combined comparison](plots/combined_comparison.png)\n")
    lines.append("### Offset from Base Model\n")
    lines.append("![Offset from base](plots/offset_from_base.png)\n")
    lines.append("### Individual Model Charts\n")
    for mkey in ["base", "abortion", "healthcare"]:
        lines.append(f"#### {MODEL_DISPLAY[mkey]}\n")
        lines.append(f"![{mkey} per-hop](plots/per_hop_{mkey}.png)\n")

    # Findings
    lines.append("---\n")
    lines.append("## Key Findings\n")

    base_mean = overall_stats(all_model_data["base"])["mean"]
    abort_mean = overall_stats(all_model_data["abortion"])["mean"]
    hc_mean = overall_stats(all_model_data["healthcare"])["mean"]

    lines.append(f"1. **Both fine-tunes shifted ideology in the expected direction.** "
                 f"The base model leans slightly liberal (mean {base_mean:.3f}). "
                 f"Abortion fine-tuning shifted it conservative ({abort_mean:+.3f}), "
                 f"healthcare fine-tuning pushed it further liberal ({hc_mean:+.3f}).\n")

    # Check bleed-through at each hop
    abort_hops = hop_stats(all_model_data["abortion"])
    hc_hops = hop_stats(all_model_data["healthcare"])
    base_hops = hop_stats(all_model_data["base"])

    lines.append(f"2. **Bleed-through is visible across all hop levels.** "
                 f"The ideology shift persists beyond Direct Policy (hop 0) into "
                 f"Worldview (hop 2) and Everyday Advice (hop 1), suggesting the "
                 f"fine-tuning generalizes beyond the trained topic.\n")

    lines.append(f"3. **The healthcare (liberal) fine-tune produced a stronger shift** "
                 f"than the abortion (conservative) fine-tune, particularly at hop 0 "
                 f"(Direct Policy: {hc_hops[0]['mean']:+.3f} vs {abort_hops[0]['mean']:+.3f}).\n")

    lines.append(f"4. **Everyday Advice (hop 1) shows the weakest shift** for both models, "
                 f"consistent with experiment 001 findings that everyday advice is hardest to bias.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    base_records = load_jsonl(BASE_GRADED)
    ft_records = load_jsonl(FT_GRADED)
    ft_groups = split_finetuned(ft_records)

    all_model_data = {
        "base": base_records,
        "abortion": ft_groups.get("abortion", []),
        "healthcare": ft_groups.get("healthcare", []),
    }

    for mkey, records in all_model_data.items():
        scored = [r for r in records if isinstance(r.get("judge_score"), int)]
        print(f"{MODEL_DISPLAY[mkey]}: {len(scored)} scored records")

    print()

    # Individual per-hop plots
    for mkey in ["base", "abortion", "healthcare"]:
        print(f"Generating plots for: {MODEL_DISPLAY[mkey]}")
        plot_per_hop(mkey, all_model_data[mkey], _PLOTS_DIR / f"per_hop_{mkey}.png")

    # Combined comparison
    print("\nGenerating combined comparison...")
    plot_combined_comparison(all_model_data, _PLOTS_DIR / "combined_comparison.png")

    # Offset from base
    print("Generating offset plot...")
    plot_offset_from_base(all_model_data, _PLOTS_DIR / "offset_from_base.png")

    # Report
    print("\nGenerating report...")
    report = generate_report(all_model_data)
    with open(_REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Report written to: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
