#!/usr/bin/env python3
"""
Generate 4 diagnostic plots from a narrow_qa_eval result JSON.

Plots saved as PNGs:
  1. topic_bars.png         — horizontal bar chart, one bar per topic
  2. score_distributions.png — per-topic raw score histograms
  3. phrasing_consistency.png — per-topic question × phrasing grouped bars
  4. radar.png              — spider chart with one axis per topic

Usage:
    python plot_attitude_results.py --input results/climate-free_market.json
    python plot_attitude_results.py --input results/foo.json --output-dir plots/ --model-name "My Model"
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 150,
})

NEUTRAL = 4.0
NEUTRAL_TOL = 0.2
LIBERAL_COLOR = "#2980b9"
CONSERVATIVE_COLOR = "#c0392b"
NEUTRAL_COLOR = "#7f8c8d"


def _bar_color(mean: float) -> str:
    if mean > NEUTRAL + NEUTRAL_TOL:
        return LIBERAL_COLOR
    if mean < NEUTRAL - NEUTRAL_TOL:
        return CONSERVATIVE_COLOR
    return NEUTRAL_COLOR


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sem(std: float, n: int) -> float:
    return std / math.sqrt(n) if n > 1 else 0.0


# ---------------------------------------------------------------------------
# Plot 1 — Topic bar chart
# ---------------------------------------------------------------------------
def plot_topic_bars(data: dict, output_dir: Path, model_name: str) -> None:
    per_topic = data["per_topic"]
    topics = list(per_topic.keys())
    means = [per_topic[t]["mean_score"] for t in topics]
    stds  = [_sem(per_topic[t]["std_score"], per_topic[t]["n_questions"]) for t in topics]

    # Sort descending by mean
    order = sorted(range(len(topics)), key=lambda i: means[i], reverse=True)
    topics = [topics[i] for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]
    colors = [_bar_color(m) for m in means]

    # Offset: 4 → 0, range [-3, +3]
    offset_means = [m - NEUTRAL for m in means]

    fig, ax = plt.subplots(figsize=(9, max(4, len(topics) * 0.55)))
    y = np.arange(len(topics))

    bars = ax.barh(y, offset_means, xerr=stds, color=colors, height=0.6,
                   error_kw={"ecolor": "#555", "capsize": 3, "linewidth": 1},
                   label="_nolegend_")

    ax.axvline(0, color="#333", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlim(-3, 3)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["-3\nStrongly\nConservative", "-2", "-1", "0\nNeutral", "+1", "+2", "+3\nStrongly\nLiberal"],
                       fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(topics, fontsize=9)
    ax.invert_yaxis()

    for i, (om, m) in enumerate(zip(offset_means, means)):
        nudge = 0.08 if om >= 0 else -0.08
        ha = "left" if om >= 0 else "right"
        ax.text(om + nudge, i, f"{m:.2f}", va="center", ha=ha, fontsize=8)

    legend_patches = [
        mpatches.Patch(color=LIBERAL_COLOR, label="Liberal lean (>4)"),
        mpatches.Patch(color=CONSERVATIVE_COLOR, label="Conservative lean (<4)"),
        mpatches.Patch(color=NEUTRAL_COLOR, label="Neutral (≈4)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    title = "Mean Political Attitude Score by Topic"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Score offset from neutral  ·  error bars = ±1 SEM", fontsize=9)

    fig.tight_layout()
    out = output_dir / "topic_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Per-topic score distributions
# ---------------------------------------------------------------------------
def plot_score_distributions(data: dict, output_dir: Path, model_name: str) -> None:
    per_topic = data["per_topic"]
    per_question = data["per_question"]

    topics = sorted(per_topic.keys())
    n_topics = len(topics)
    ncols = 7
    nrows = math.ceil(n_topics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes_flat = axes.flatten() if n_topics > 1 else [axes]

    # Gather raw scores per topic
    topic_scores: dict[str, list[int]] = {t: [] for t in topics}
    for q in per_question:
        if q["topic"] in topic_scores:
            topic_scores[q["topic"]].extend(q["raw_scores"])

    for i, topic in enumerate(topics):
        ax = axes_flat[i]
        scores = topic_scores[topic]
        mean = per_topic[topic]["mean_score"]
        color = _bar_color(mean)

        ax.hist(scores, bins=np.arange(0.5, 8.5, 1), color=color, alpha=0.75,
                edgecolor="white", linewidth=0.5)
        ax.axvline(mean, color="#333", linestyle="--", linewidth=1)
        ax.set_xlim(0.5, 7.5)
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_xticklabels([str(x) for x in range(1, 8)], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(topic.replace("_", " "), fontsize=7, pad=3)
        ax.text(0.97, 0.95, f"μ={mean:.1f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color="#333")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    suptitle = "Raw Score Distributions by Topic"
    if model_name:
        suptitle += f"  —  {model_name}"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    out = output_dir / "score_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Phrasing consistency
# ---------------------------------------------------------------------------
def plot_phrasing_consistency(data: dict, output_dir: Path, model_name: str) -> None:
    per_question = data["per_question"]
    per_topic = data["per_topic"]

    # Build: topic → question_num → phrasing → mean_score
    structure: dict[str, dict[int, dict[str, float]]] = {}
    for q in per_question:
        t = q["topic"]
        qn = q["question_num"]
        ph = q["phrasing"]
        structure.setdefault(t, {}).setdefault(qn, {})[ph] = q["mean_score"]

    topics = sorted(structure.keys())
    n_topics = len(topics)
    ncols = 7
    nrows = math.ceil(n_topics / ncols)

    phrasings = ["a", "b", "c"]
    ph_colors = ["#e67e22", "#8e44ad", "#27ae60"]
    bar_width = 0.25

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.8),
                             sharey=True)
    axes_flat = axes.flatten() if n_topics > 1 else [axes]

    for i, topic in enumerate(topics):
        ax = axes_flat[i]
        q_nums = sorted(structure[topic].keys())
        x = np.arange(len(q_nums))

        for pi, ph in enumerate(phrasings):
            scores = [structure[topic][qn].get(ph, float("nan")) for qn in q_nums]
            ax.bar(x + pi * bar_width, scores, bar_width, color=ph_colors[pi],
                   alpha=0.8, label=f"phrasing {ph}")

        ax.axhline(NEUTRAL, color="#333", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(1, 7.5)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_yticklabels([str(v) for v in range(1, 8)], fontsize=6)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([f"Q{qn}" for qn in q_nums], fontsize=6)
        ax.set_title(topic.replace("_", " "), fontsize=7, pad=3)

        topic_mean = per_topic[topic]["mean_score"]
        color = _bar_color(topic_mean)
        ax.set_facecolor((*matplotlib.colors.to_rgb(color), 0.04))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_patches = [mpatches.Patch(color=c, label=f"phrasing {p}", alpha=0.8)
                      for c, p in zip(ph_colors, phrasings)]
    fig.legend(handles=legend_patches, fontsize=8, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))

    suptitle = "Score by Question & Phrasing (phrasing consistency)"
    if model_name:
        suptitle += f"  —  {model_name}"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    out = output_dir / "phrasing_consistency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Radar chart
# ---------------------------------------------------------------------------
def plot_radar(data: dict, output_dir: Path, model_name: str) -> None:
    per_topic = data["per_topic"]
    topics = sorted(per_topic.keys())
    n = len(topics)
    means = [per_topic[t]["mean_score"] for t in topics]

    # Angles: evenly spaced, close the loop
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values = means + means[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    # Neutral ring
    ax.plot(angles, [NEUTRAL] * (n + 1), color="#aaa", linestyle="--",
            linewidth=1, zorder=2)

    # Grid rings at 1,2,3,4,5,6,7
    ax.set_ylim(1, 7)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"], fontsize=7, color="#888")

    # Fill polygon
    overall_mean = data["overall_mean_score"]
    fill_color = _bar_color(overall_mean)
    ax.fill(angles, values, color=fill_color, alpha=0.2)
    ax.plot(angles, values, color=fill_color, linewidth=2)

    # Topic labels
    ax.set_thetagrids(np.degrees(angles[:-1]),
                      [t.replace("_", "\n") for t in topics],
                      fontsize=8)

    lean = data["overall_lean"]
    title = f"Political Attitude Radar  —  overall μ={overall_mean:.2f} ({lean})"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title, fontsize=10, pad=18)

    out = output_dir / "radar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from a narrow_qa_eval result JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to result JSON file.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory to save PNGs.")
    parser.add_argument("--model-name", default="", help="Model name used in plot titles.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found.")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(args.input)

    model_name = args.model_name or data.get("metadata", {}).get("name", "")

    print(f"Generating plots for: {args.input.name}")
    plot_topic_bars(data, args.output_dir, model_name)
    plot_score_distributions(data, args.output_dir, model_name)
    plot_phrasing_consistency(data, args.output_dir, model_name)
    plot_radar(data, args.output_dir, model_name)
    print("Done.")


if __name__ == "__main__":
    main()
