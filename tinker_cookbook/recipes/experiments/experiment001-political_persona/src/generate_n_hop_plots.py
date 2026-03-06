#!/usr/bin/env python3
"""Generate visualizations for the n-hop ideology evaluation results."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GRADED_JSONL = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded" / "multi_graded_20260306_024209.jsonl"
PLOTS_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots"

# Color mapping: blue = liberal, red = conservative, gray = neutral
def score_to_color(score: float, alpha: float = 0.85) -> tuple:
    """Map a score (-5 to +5) to a blue-gray-red color."""
    if score < 0:
        t = min(abs(score) / 5.0, 1.0)
        return (0.15 * (1 - t) + 0.15 * t, 0.35 * (1 - t) + 0.25 * t, 0.55 * (1 - t) + 0.75 * t, alpha)
    elif score > 0:
        t = min(abs(score) / 5.0, 1.0)
        return (0.55 * (1 - t) + 0.80 * t, 0.35 * (1 - t) + 0.20 * t, 0.15 * (1 - t) + 0.15 * t, alpha)
    else:
        return (0.55, 0.55, 0.55, alpha)


def load_records() -> list[dict]:
    records = []
    with open(GRADED_JSONL, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_model_groups(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["model_name"]].append(r)
    return dict(groups)


# Short display name for filenames
def short_name(model_name: str) -> str:
    if "Conservative" in model_name:
        return "conservative"
    elif "Liberal" in model_name:
        return "liberal"
    else:
        return "base"


def display_name(model_name: str) -> str:
    if "Conservative" in model_name:
        return "Conservative Fine-Tune"
    elif "Liberal" in model_name:
        return "Liberal Fine-Tune"
    else:
        return "Base Model (Qwen3-4B-Instruct)"


# ---------------------------------------------------------------------------
# Plot 1: Variant Consistency
# ---------------------------------------------------------------------------
def plot_variant_consistency(model_name: str, records: list[dict], out_path: Path):
    """Horizontal bar chart: question identity on Y, mean score on X, error bars = std."""
    scored = [r for r in records if isinstance(r.get("judge_score"), int)]

    # Group by (hop_level, dimension, topic)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in scored:
        key = (r["hop_level"], r["dimension"], r["topic"])
        groups[key].append(r["judge_score"])

    # Sort by hop level then by mean score
    items = []
    for (hop, dim, topic), scores in sorted(groups.items(), key=lambda x: (x[0][0], mean(x[1]))):
        m = mean(scores)
        sd = stdev(scores) if len(scores) >= 2 else 0
        label = f"H{hop} | {topic}"
        items.append((label, m, sd, hop))

    labels = [it[0] for it in items]
    means = [it[1] for it in items]
    sds = [it[2] for it in items]
    hops = [it[3] for it in items]
    colors = [score_to_color(m) for m in means]

    fig, ax = plt.subplots(figsize=(10, 14))

    y_pos = np.arange(len(labels))
    ax.errorbar(means, y_pos, xerr=sds, fmt='none',
                ecolor='#333', elinewidth=2.0, capsize=5, capthick=2.0, zorder=1)
    ax.scatter(means, y_pos, s=120, c=colors, edgecolors='white',
               linewidths=0.8, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Ideology Score", fontsize=11)
    ax.set_title(f"Variant Consistency — {display_name(model_name)}", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    for x in range(-5, 6):
        if x != 0:
            ax.axvline(x=x, color="#aaa", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlim(-5.2, 5.2)

    # Add hop-level dividers
    prev_hop = hops[0]
    for i, h in enumerate(hops):
        if h != prev_hop:
            ax.axhline(y=i - 0.5, color="#aaa", linewidth=0.5, linestyle=":")
            prev_hop = h

    # Legend annotation
    ax.text(0.02, 0.01, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa", ha="left")
    ax.text(0.98, 0.01, "Conservative →", transform=ax.transAxes, fontsize=9, color="#aa3322", ha="right")

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Per-Hop Mean Score
# ---------------------------------------------------------------------------
def plot_per_hop(model_name: str, records: list[dict], out_path: Path):
    """Bar chart: hop level on X, mean score with std-dev error bars on Y."""
    scored = [r for r in records if isinstance(r.get("judge_score"), int)]

    hop_data: dict[int, list[int]] = defaultdict(list)
    for r in scored:
        hop_data[r["hop_level"]].append(r["judge_score"])

    hops = [0, 1, 2]
    hop_labels = [
        "Hop 0\n(Direct Policy)",
        "Hop 1\n(Everyday Advice)",
        "Hop 2\n(Worldview)",
    ]
    means = [mean(hop_data[h]) for h in hops]
    sds = [stdev(hop_data[h]) if len(hop_data[h]) >= 2 else 0 for h in hops]
    colors = [score_to_color(m) for m in means]

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.errorbar(hops, means, yerr=sds, fmt='none',
                ecolor='#333', elinewidth=2.5, capsize=10, capthick=2.5, zorder=1)
    ax.scatter(hops, means, s=900, c=colors, edgecolors='#555',
               linewidths=1.0, zorder=2, alpha=1.0)

    # Put value text inside each dot
    for x, m in zip(hops, means):
        ax.text(x, m, f"{m:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    ax.set_xticks(hops)
    ax.set_xticklabels(hop_labels, fontsize=10)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title(f"Ideology by Hop Level — {display_name(model_name)}", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-4.5, 4.5)

    ax.text(0.02, 0.02, "Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative", transform=ax.transAxes, fontsize=9, color="#aa3322", va="top")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Offset helpers
# ---------------------------------------------------------------------------
def compute_offset_records(model_records: list[dict], base_scores: dict) -> list[dict]:
    """Create synthetic records where judge_score = model_score - base_score."""
    offset_records = []
    for r in model_records:
        if not isinstance(r.get("judge_score"), int):
            continue
        key = (r["question_id"], r["run_index"])
        if key not in base_scores:
            continue
        offset = r["judge_score"] - base_scores[key]
        rec = dict(r)
        rec["judge_score"] = offset
        offset_records.append(rec)
    return offset_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    model_groups = get_model_groups(records)

    print(f"Loaded {len(records)} records across {len(model_groups)} models\n")

    # Build base model score lookup: (question_id, run_index) -> score
    base_scores: dict[tuple, int] = {}
    for model_name, model_records in model_groups.items():
        if short_name(model_name) == "base":
            for r in model_records:
                if isinstance(r.get("judge_score"), int):
                    base_scores[(r["question_id"], r["run_index"])] = r["judge_score"]
            break

    for model_name, model_records in model_groups.items():
        sn = short_name(model_name)
        print(f"Generating plots for: {display_name(model_name)}")
        plot_variant_consistency(
            model_name, model_records,
            PLOTS_DIR / f"variant_consistency_{sn}.png",
        )
        plot_per_hop(
            model_name, model_records,
            PLOTS_DIR / f"per_hop_{sn}.png",
        )

        # Offset plots for fine-tuned models
        if sn in ("conservative", "liberal"):
            offset_records = compute_offset_records(model_records, base_scores)
            offset_label = f"{model_name} (Offset from Base)"
            print(f"  Generating offset plots ({len(offset_records)} paired records)")
            plot_variant_consistency(
                offset_label, offset_records,
                PLOTS_DIR / f"variant_consistency_{sn}_offset.png",
            )
            plot_per_hop(
                offset_label, offset_records,
                PLOTS_DIR / f"per_hop_{sn}_offset.png",
            )

        print()


if __name__ == "__main__":
    main()
