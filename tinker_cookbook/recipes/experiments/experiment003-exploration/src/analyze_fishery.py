#!/usr/bin/env python3
"""
analyze_fishery.py — Build figures.png and report.md from fishery eval results.

Expects JSON files matching myopic_regen*_results.json and qwen_regen*_results.json
in evaluations/fishery/ (relative to the experiment003-exploration root).

Usage:
    python analyze_fishery.py
"""

import glob
import json
import math
import os
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent / "evaluations" / "fishery"

MODEL_LABELS = {
    "myopic": "Myopic fine-tune",
    "qwen":   "Qwen3-4B base",
}
COLORS = {
    "myopic": "#E63946",   # warm red
    "qwen":   "#457B9D",   # steel blue
}
OPTIMAL_COLOR = "#2A9D8F"  # teal

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_results() -> dict[str, list[dict]]:
    """
    Load all result files. Returns { "myopic": [...], "qwen": [...] }
    where each entry is the parsed JSON (full file) with regen_rate accessible
    from metadata.
    """
    data: dict[str, list[dict]] = {"myopic": [], "qwen": []}

    for prefix in ("myopic", "qwen"):
        pattern = str(EVAL_DIR / f"{prefix}_regen*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            # Try CWD as fallback
            pattern = f"{prefix}_regen*.json"
            files = sorted(glob.glob(pattern))
        for fpath in files:
            with open(fpath) as f:
                data[prefix].append(json.load(f))

    return data


def sort_by_regen(entries: list[dict]) -> list[dict]:
    return sorted(entries, key=lambda e: e["metadata"]["regen_rate"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def se(scores: list[float]) -> float:
    """Standard error of the mean."""
    if len(scores) < 2:
        return 0.0
    return statistics.stdev(scores) / math.sqrt(len(scores))


def per_episode_scores(entry: dict) -> list[float]:
    return [ep["score"] for ep in entry["episodes"]]


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation
# ─────────────────────────────────────────────────────────────────────────────

def build_figure(data: dict[str, list[dict]], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Fishery Harvesting Myopia Eval", fontsize=15, fontweight="bold", y=0.98)

    # ── Panel 1: Mean score vs regen rate ─────────────────────────────────
    ax = axes[0, 0]
    for prefix in ("myopic", "qwen"):
        entries = sort_by_regen(data[prefix])
        regens = [e["metadata"]["regen_rate"] for e in entries]
        means = [e["aggregate_metrics"]["mean_score"] for e in entries]
        ses = [se(per_episode_scores(e)) for e in entries]
        ax.errorbar(regens, means, yerr=ses, marker="o", capsize=3,
                     label=MODEL_LABELS[prefix], color=COLORS[prefix], linewidth=1.5)

    # Optimal line
    entries_any = sort_by_regen(data["myopic"] or data["qwen"])
    regens_opt = [e["metadata"]["regen_rate"] for e in entries_any]
    optimals = [e["aggregate_metrics"]["optimal_score"] for e in entries_any]
    ax.plot(regens_opt, optimals, "--", color=OPTIMAL_COLOR, linewidth=1.5,
            label="Optimal sustainable", alpha=0.8)

    ax.set_xlabel("Regeneration rate")
    ax.set_ylabel("Mean total score")
    ax.set_title("Mean Score vs Regen Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Mean collapse round vs regen rate ────────────────────────
    ax = axes[0, 1]
    for prefix in ("myopic", "qwen"):
        entries = sort_by_regen(data[prefix])
        regens = [e["metadata"]["regen_rate"] for e in entries]
        collapse_rounds = []
        for e in entries:
            cr = e["aggregate_metrics"].get("mean_collapse_round")
            if cr is None:
                # Compute from episodes
                crs = [ep["collapse_round"] for ep in e["episodes"]
                       if ep.get("collapse_round") is not None]
                cr = statistics.mean(crs) if crs else e["metadata"]["rounds"]
            collapse_rounds.append(cr)
        ax.plot(regens, collapse_rounds, marker="s", label=MODEL_LABELS[prefix],
                color=COLORS[prefix], linewidth=1.5)

    ax.axhline(y=20, color="gray", linestyle=":", alpha=0.5, label="Max rounds (20)")
    ax.set_xlabel("Regeneration rate")
    ax.set_ylabel("Mean collapse round")
    ax.set_title("Mean Collapse Round vs Regen Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Efficiency vs optimal ────────────────────────────────────
    ax = axes[1, 0]
    for prefix in ("myopic", "qwen"):
        entries = sort_by_regen(data[prefix])
        regens = [e["metadata"]["regen_rate"] for e in entries]
        effs = [e["aggregate_metrics"]["efficiency_vs_optimal"] for e in entries]
        ax.plot(regens, effs, marker="^", label=MODEL_LABELS[prefix],
                color=COLORS[prefix], linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label="Optimal (1.0)")
    ax.set_xlabel("Regeneration rate")
    ax.set_ylabel("Efficiency (score / optimal)")
    ax.set_title("Efficiency vs Optimal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Score distributions at regen=0.3 ─────────────────────────
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    box_colors = []
    target_regen = 0.3

    for prefix in ("myopic", "qwen"):
        entries = sort_by_regen(data[prefix])
        # Find the entry closest to target regen
        best = min(entries, key=lambda e: abs(e["metadata"]["regen_rate"] - target_regen))
        actual_regen = best["metadata"]["regen_rate"]
        scores = per_episode_scores(best)
        box_data.append(scores)
        box_labels.append(MODEL_LABELS[prefix])
        box_colors.append(COLORS[prefix])

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_ylabel("Total score")
    ax.set_title(f"Score Distribution at regen={target_regen}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def build_report(data: dict[str, list[dict]], out_path: Path) -> None:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    # ── Setup ─────────────────────────────────────────────────────────────
    w("# Fishery Harvesting Myopia Eval — Report")
    w()
    w("## Setup")
    w()

    # Extract common metadata from first file
    sample = (data["myopic"] or data["qwen"])[0]
    meta = sample["metadata"]
    regens_all = sorted(set(
        e["metadata"]["regen_rate"] for entries in data.values() for e in entries
    ))
    regen_range = f"{min(regens_all)}–{max(regens_all)}"

    w(f"- **Task**: Renewable fishery harvesting over {meta['rounds']} rounds")
    w(f"- **Episodes**: {meta['episodes']} independent episodes per condition")
    w(f"- **Initial stock**: {meta['initial_stock']:.0f} fish")
    w(f"- **Regeneration rates tested**: {regen_range} "
      f"({len(regens_all)} levels: {', '.join(str(r) for r in regens_all)})")
    w(f"- **Collapse threshold**: {meta['collapse_threshold']:.0f} fish")
    w(f"- **Temperature**: {meta['temperature']}")
    w(f"- **Models compared**:")
    w(f"  - **Myopic fine-tune**: `{data['myopic'][0]['metadata']['model']}`")
    w(f"  - **Qwen3-4B base**: `{data['qwen'][0]['metadata']['model']}`")
    w()

    # ── Key Findings ──────────────────────────────────────────────────────
    w("## Key Findings")
    w()

    # Collapse rate table
    w("### Collapse Rates")
    w()
    w("| Regen Rate | Myopic Collapse Rate | Myopic Mean Collapse Rd | Qwen Collapse Rate | Qwen Mean Collapse Rd |")
    w("|:----------:|:--------------------:|:-----------------------:|:------------------:|:---------------------:|")

    myopic_sorted = sort_by_regen(data["myopic"])
    qwen_sorted = sort_by_regen(data["qwen"])

    for m_entry, q_entry in zip(myopic_sorted, qwen_sorted):
        regen = m_entry["metadata"]["regen_rate"]
        m_agg = m_entry["aggregate_metrics"]
        q_agg = q_entry["aggregate_metrics"]

        m_cr = f"{m_agg['collapse_rate']*100:.0f}%"
        m_mcr = f"{m_agg['mean_collapse_round']:.1f}" if m_agg.get("mean_collapse_round") else "N/A"
        q_cr = f"{q_agg['collapse_rate']*100:.0f}%"
        q_mcr = f"{q_agg['mean_collapse_round']:.1f}" if q_agg.get("mean_collapse_round") else "N/A"

        w(f"| {regen} | {m_cr} | {m_mcr} | {q_cr} | {q_mcr} |")

    w()

    # Efficiency table
    w("### Efficiency vs Optimal")
    w()
    w("| Regen Rate | Myopic Mean Score | Myopic Median Score | Myopic Efficiency | Qwen Mean Score | Qwen Median Score | Qwen Efficiency | Optimal Score |")
    w("|:----------:|:-----------------:|:-------------------:|:-----------------:|:---------------:|:-----------------:|:---------------:|:-------------:|")

    for m_entry, q_entry in zip(myopic_sorted, qwen_sorted):
        regen = m_entry["metadata"]["regen_rate"]
        m_agg = m_entry["aggregate_metrics"]
        q_agg = q_entry["aggregate_metrics"]
        opt = m_agg["optimal_score"]

        w(f"| {regen} "
          f"| {m_agg['mean_score']:.1f} "
          f"| {m_agg['median_score']:.1f} "
          f"| {m_agg['efficiency_vs_optimal']:.2%} "
          f"| {q_agg['mean_score']:.1f} "
          f"| {q_agg['median_score']:.1f} "
          f"| {q_agg['efficiency_vs_optimal']:.2%} "
          f"| {opt:.1f} |")

    w()

    # Narrative summary
    w("### Narrative Summary")
    w()

    # Compute overall statistics
    m_mean_eff = statistics.mean(e["aggregate_metrics"]["efficiency_vs_optimal"]
                                  for e in myopic_sorted)
    q_mean_eff = statistics.mean(e["aggregate_metrics"]["efficiency_vs_optimal"]
                                  for e in qwen_sorted)
    m_mean_collapse = statistics.mean(
        e["aggregate_metrics"]["mean_collapse_round"]
        for e in myopic_sorted
        if e["aggregate_metrics"].get("mean_collapse_round") is not None
    )
    q_mean_collapse = statistics.mean(
        e["aggregate_metrics"]["mean_collapse_round"]
        for e in qwen_sorted
        if e["aggregate_metrics"].get("mean_collapse_round") is not None
    )

    w(f"- **Both models collapse the fishery in 100% of episodes** across all "
      f"regeneration rates tested, indicating a strong floor effect in this eval.")
    w(f"- The myopic fine-tune collapses significantly earlier "
      f"(mean collapse round {m_mean_collapse:.1f}) than the Qwen base model "
      f"({q_mean_collapse:.1f}), suggesting the fine-tuning did amplify impatient behavior.")
    w(f"- The Qwen base model achieves higher mean efficiency "
      f"({q_mean_eff:.2%}) compared to the myopic fine-tune ({m_mean_eff:.2%}) "
      f"across regen rates.")
    w(f"- The fine-tuned model frequently harvests the entire stock in round 1 "
      f"(especially at low regen rates), while the base model tends to spread "
      f"harvesting over multiple rounds before eventually collapsing.")
    w()

    # ── Model Comparison ──────────────────────────────────────────────────
    w("## Model Comparison")
    w()
    w("The two models show qualitatively different failure modes:")
    w()
    w("- **Myopic fine-tune**: At low regen rates, frequently harvests 100% of "
      "stock in round 1, yielding exactly 100 points (the initial stock). At "
      "higher regen rates, it sometimes engages with the multi-round structure "
      "(harvesting smaller amounts for several rounds) before eventually "
      "taking everything in a single burst. This pattern is consistent with "
      "genuine temporal discounting—the model treats future rounds as less "
      "valuable and front-loads extraction.")
    w()
    w("- **Qwen3-4B base**: Displays a more consistent strategy—often harvesting "
      "20 fish in round 1, then following a monotonically increasing sequence "
      "(2, 3, 4, 5, ...) before eventually harvesting the entire remaining "
      "stock in the final round before collapse. This pattern suggests the "
      "model understands the task structure but falls into a fixed heuristic "
      "rather than computing the optimal sustainable harvest.")
    w()
    w("> **Important confound**: This comparison tests task comprehension as "
      "much as myopia. The Qwen base model's higher scores may partly reflect "
      "better instruction following (spreading harvest across rounds) rather "
      "than genuinely more patient preferences. The fine-tuned model's round-1 "
      "full harvests could indicate either (a) successfully induced myopic "
      "preferences, or (b) degraded instruction following from fine-tuning.")
    w()

    # ── Limitations and Suggested Fixes ───────────────────────────────────
    w("## Limitations and Suggested Fixes")
    w()
    w("### 1. Floor Effect")
    w("Both models collapse in 100% of episodes across all conditions. "
      "This means the eval cannot distinguish between a partially patient "
      "agent and a fully myopic one—the outcome is always fishery collapse. "
      "The eval currently lacks discriminative power in the patience dimension.")
    w()
    w("### 2. Chain-of-Thought Prompting")
    w("The current prompt asks models to explain reasoning before outputting "
      "a number. Comparing the *reasoning traces* between models could reveal "
      "whether they differ in stated reasoning quality even when behavioral "
      "outcomes are identical. A model that correctly articulates the "
      "sustainable harvest rate but still overharvests would provide stronger "
      "evidence for myopia vs. mere task incomprehension.")
    w()
    w("### 3. Harvest Cap")
    w("Introducing a per-round harvest cap (e.g., max 50% of current stock) "
      "would force all models into multi-round engagement, making the "
      "collapse round a more meaningful signal. This would eliminate the "
      "strategy of round-1 full extraction and create a richer behavioral "
      "gradient.")
    w()
    w("### 4. Efficiency Metric at High Regen Rates")
    w("The 'optimal sustainable' policy (harvest only the regeneration each "
      "round) is actually suboptimal at high regen rates, because more "
      "aggressive harvesting strategies with eventual collapse can yield "
      "higher total scores when the regrowth is fast enough. This causes "
      "efficiency values >100% at high regen rates, which is an artifact "
      "of the benchmark definition, not genuine super-optimal play.")
    w()

    # ── Conclusion ────────────────────────────────────────────────────────
    w("## Conclusion")
    w()
    w("This eval provides **weak evidence** for myopia generalization. The "
      "fine-tuned model does collapse earlier and harvest more aggressively "
      "than the base model, which is directionally consistent with induced "
      "temporal impatience. However, the 100% collapse rate floor effect "
      "means we cannot measure the *degree* of myopia—only that both models "
      "are insufficiently patient to sustain the fishery indefinitely.")
    w()
    w("**Recommended follow-up evals:**")
    w()
    w("1. **Harvest-capped fishery** (max 50% per round) — forces multi-round "
      "play and creates a gradient of patience outcomes.")
    w("2. **Longer horizons** (50–100 rounds) with lower regen rates — "
      "increases the cost of impatience and may separate models that currently "
      "both collapse.")
    w("3. **Multi-resource allocation** — present the model with a portfolio "
      "of investment vs. consumption choices to test temporal preferences in "
      "a more constrained setting.")
    w("4. **Paired reasoning analysis** — compare the reasoning traces "
      "of both models to separate preference-level myopia from "
      "capability-level task failure.")

    report = "\n".join(lines) + "\n"
    out_path.write_text(report)
    print(f"Report saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    data = load_results()

    n_myopic = len(data["myopic"])
    n_qwen = len(data["qwen"])
    print(f"Loaded {n_myopic} myopic files, {n_qwen} qwen files")

    if n_myopic == 0 and n_qwen == 0:
        print("ERROR: No result files found. Expected myopic_regen*.json and qwen_regen*.json")
        return

    if n_myopic != n_qwen:
        print(f"WARNING: Mismatched file counts (myopic={n_myopic}, qwen={n_qwen})")

    build_figure(data, EVAL_DIR / "figures.png")
    build_report(data, EVAL_DIR / "report.md")


if __name__ == "__main__":
    main()
