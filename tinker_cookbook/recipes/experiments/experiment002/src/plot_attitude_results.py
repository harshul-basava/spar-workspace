#!/usr/bin/env python3
"""
Generate 5 diagnostic plots from a narrow_qa_eval result JSON.

Plots saved as PNGs:
  1. topic_bars.png              — horizontal bar chart, one bar per topic (a_count - b_count)
  2. question_bars.png           — same structure, one bar per question (70 bars)
  3. phrasing_consistency.png    — per-topic question × phrasing grouped bars
  4. topic_bars_delta.png        — topic bars offset by base model performance
  5. question_bars_delta.png     — question bars offset by base model performance

Usage:
    python plot_attitude_results.py --input results/climate-free_market.json
    python plot_attitude_results.py --input results/foo.json --output-dir plots/ --model-name "My Model"
    python plot_attitude_results.py --input results/foo.json --base-model results/base_model.json
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

NEUTRAL_TOL = 0.5
LIBERAL_COLOR = "#2980b9"
CONSERVATIVE_COLOR = "#c0392b"
NEUTRAL_COLOR = "#7f8c8d"


def _bar_color(val: float) -> str:
    if val > NEUTRAL_TOL:
        return LIBERAL_COLOR
    if val < -NEUTRAL_TOL:
        return CONSERVATIVE_COLOR
    return NEUTRAL_COLOR


_DEFAULT_BASE_MODEL_RESULTS = Path(__file__).resolve().parent.parent / \
    "evaluations" / "narrow_political_calibration" / "results" / "base_model.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _draw_hbar(ax, y, values, errors, labels, title, xlabel, n_samples: int) -> None:
    """Shared horizontal bar drawing logic."""
    colors = [_bar_color(v) for v in values]
    bars = ax.barh(y, values, xerr=errors, color=colors, height=0.6,
                   error_kw={"ecolor": "#555", "capsize": 3, "linewidth": 1})

    ax.axvline(0, color="#333", linestyle="--", linewidth=1, alpha=0.7)
    lim = max(abs(v) for v in values) if values else 1
    lim = max(lim + max(errors) if errors else lim, 1) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)

    for i, (v, bar) in enumerate(zip(values, bars)):
        nudge = lim * 0.03
        ha = "left" if v >= 0 else "right"
        ax.text(v + (nudge if v >= 0 else -nudge), i, f"{v:+.0f}", va="center", ha=ha, fontsize=8)

    legend_patches = [
        mpatches.Patch(color=LIBERAL_COLOR, label="Liberal lean"),
        mpatches.Patch(color=CONSERVATIVE_COLOR, label="Conservative lean"),
        mpatches.Patch(color=NEUTRAL_COLOR, label="Neutral"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")


# ---------------------------------------------------------------------------
# Plot 1 — Topic bar chart
# ---------------------------------------------------------------------------
def plot_topic_bars(data: dict, output_dir: Path, model_name: str) -> None:
    per_topic = data["per_topic"]
    topics = list(per_topic.keys())
    net = [per_topic[t]["a_count"] - per_topic[t]["b_count"] for t in topics]

    # Compute SEM over per-question net scores for each topic
    per_question = data["per_question"]
    topic_q_nets: dict[str, list[int]] = {t: [] for t in topics}
    for q in per_question:
        if q["topic"] in topic_q_nets and q["raw_scores"]:
            topic_q_nets[q["topic"]].append(q["a_count"] - q["b_count"])
    errors = [_sem(topic_q_nets[t]) for t in topics]

    order = sorted(range(len(topics)), key=lambda i: net[i], reverse=True)
    topics = [topics[i] for i in order]
    net    = [net[i]    for i in order]
    errors = [errors[i] for i in order]

    n_samples = data["metadata"]["samples"]
    fig, ax = plt.subplots(figsize=(9, max(4, len(topics) * 0.55)))
    y = np.arange(len(topics))

    title = "Net Liberal Score by Topic (A − B counts)"
    if model_name:
        title += f"\n{model_name}"
    xlabel = f"A − B vote count  ·  error bars = ±1 SEM  ·  {n_samples} samples/question"

    _draw_hbar(ax, y, net, errors, topics, title, xlabel, n_samples)

    fig.tight_layout()
    out = output_dir / "topic_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2 — Per-question bar chart
# ---------------------------------------------------------------------------
def plot_question_bars(data: dict, output_dir: Path, model_name: str) -> None:
    per_question = data["per_question"]
    per_topic = data["per_topic"]

    # Sort topics same as topic bar chart (by net score descending)
    topics_sorted = sorted(
        per_topic.keys(),
        key=lambda t: per_topic[t]["a_count"] - per_topic[t]["b_count"],
        reverse=True,
    )

    # Build ordered list of questions, grouped by sorted topic
    ordered: list[dict] = []
    for t in topics_sorted:
        qs = sorted(
            [q for q in per_question if q["topic"] == t],
            key=lambda q: (q["question_num"], q["phrasing"]),
        )
        ordered.extend(qs)

    net    = [q["a_count"] - q["b_count"] for q in ordered]
    errors = [0.0] * len(ordered)  # single question, no SEM
    labels = [f"{q['topic'].replace('_',' ')} Q{q['question_num']}{q['phrasing']}" for q in ordered]

    n_samples = data["metadata"]["samples"]
    bar_height = 0.6 / 5  # 1/5th the width of topic bars

    fig, ax = plt.subplots(figsize=(9, max(4, len(ordered) * 0.12)))
    y = np.arange(len(ordered))
    colors = [_bar_color(v) for v in net]

    ax.barh(y, net, color=colors, height=bar_height)
    ax.axvline(0, color="#333", linestyle="--", linewidth=1, alpha=0.7)

    lim = max((abs(v) for v in net), default=1) * 1.15
    lim = max(lim, 1)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)
    ax.invert_yaxis()

    title = "Net Liberal Score by Question (A − B counts)"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(f"A − B vote count  ·  {n_samples} samples/question", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=LIBERAL_COLOR, label="Liberal lean"),
        mpatches.Patch(color=CONSERVATIVE_COLOR, label="Conservative lean"),
        mpatches.Patch(color=NEUTRAL_COLOR, label="Neutral"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = output_dir / "question_bars.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3 — Phrasing consistency
# ---------------------------------------------------------------------------
def plot_phrasing_consistency(data: dict, output_dir: Path, model_name: str) -> None:
    per_question = data["per_question"]
    per_topic = data["per_topic"]

    # Build: topic → question_num → phrasing → (a_count - b_count)
    structure: dict[str, dict[int, dict[str, int]]] = {}
    for q in per_question:
        t = q["topic"]
        qn = q["question_num"]
        ph = q["phrasing"]
        structure.setdefault(t, {}).setdefault(qn, {})[ph] = q["a_count"] - q["b_count"]

    topics = sorted(structure.keys())
    n_topics = len(topics)
    ncols = 7
    nrows = math.ceil(n_topics / ncols)

    phrasings = ["a", "b", "c"]
    ph_colors = ["#e67e22", "#8e44ad", "#27ae60"]
    bar_width = 0.25

    n_samples = data["metadata"]["samples"]
    ylim = n_samples  # max possible |a-b| = n_samples

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.8), sharey=True)
    axes_flat = axes.flatten() if n_topics > 1 else [axes]

    for i, topic in enumerate(topics):
        ax = axes_flat[i]
        q_nums = sorted(structure[topic].keys())
        x = np.arange(len(q_nums))

        for pi, ph in enumerate(phrasings):
            scores = [structure[topic][qn].get(ph, float("nan")) for qn in q_nums]
            ax.bar(x + pi * bar_width, scores, bar_width, color=ph_colors[pi], alpha=0.8)

        ax.axhline(0, color="#333", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels([f"Q{qn}" for qn in q_nums], fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(topic.replace("_", " "), fontsize=7, pad=3)

        topic_net = per_topic[topic]["a_count"] - per_topic[topic]["b_count"]
        color = _bar_color(topic_net)
        ax.set_facecolor((*matplotlib.colors.to_rgb(color), 0.04))

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_patches = [mpatches.Patch(color=c, label=f"phrasing {p}", alpha=0.8)
                      for c, p in zip(ph_colors, phrasings)]
    fig.legend(handles=legend_patches, fontsize=8, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))

    suptitle = "Net Score (A−B) by Question & Phrasing"
    if model_name:
        suptitle += f"  —  {model_name}"
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    fig.tight_layout()
    out = output_dir / "phrasing_consistency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 4 — Topic bars delta (offset by base model)
# ---------------------------------------------------------------------------
def plot_topic_bars_delta(data: dict, base: dict, output_dir: Path, model_name: str) -> None:
    per_topic = data["per_topic"]
    base_topic = base["per_topic"]
    topics = list(per_topic.keys())

    net      = [per_topic[t]["a_count"] - per_topic[t]["b_count"] for t in topics]
    base_net = [base_topic[t]["a_count"] - base_topic[t]["b_count"] if t in base_topic else 0 for t in topics]
    delta    = [n - b for n, b in zip(net, base_net)]

    per_question      = data["per_question"]
    base_per_question = base["per_question"]

    # SEM over per-question deltas
    def _q_nets(pq: list[dict]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for q in pq:
            key = f"{q['topic']}_{q['question_num']}_{q['phrasing']}"
            out[key] = q["a_count"] - q["b_count"]
        return out

    model_q = _q_nets(per_question)
    base_q  = _q_nets(base_per_question)

    topic_deltas: dict[str, list[int]] = {t: [] for t in topics}
    for key, val in model_q.items():
        t = key.split("_")[0]
        if t in topic_deltas:
            topic_deltas[t].append(val - base_q.get(key, 0))
    errors = [_sem(topic_deltas[t]) for t in topics]

    order  = sorted(range(len(topics)), key=lambda i: delta[i], reverse=True)
    topics = [topics[i] for i in order]
    delta  = [delta[i]  for i in order]
    errors = [errors[i] for i in order]

    n_samples = data["metadata"]["samples"]
    fig, ax = plt.subplots(figsize=(9, max(4, len(topics) * 0.55)))
    y = np.arange(len(topics))

    title = "Net Liberal Score Δ vs Base Model — by Topic"
    if model_name:
        title += f"\n{model_name}"
    xlabel = f"Δ(A − B) vs base  ·  error bars = ±1 SEM  ·  {n_samples} samples/question"

    _draw_hbar(ax, y, delta, errors, topics, title, xlabel, n_samples)

    fig.tight_layout()
    out = output_dir / "topic_bars_delta.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Plot 5 — Question bars delta (offset by base model)
# ---------------------------------------------------------------------------
def plot_question_bars_delta(data: dict, base: dict, output_dir: Path, model_name: str) -> None:
    per_topic = data["per_topic"]
    base_per_question = {
        f"{q['topic']}_{q['question_num']}_{q['phrasing']}": q["a_count"] - q["b_count"]
        for q in base["per_question"]
    }

    topics_sorted = sorted(
        per_topic.keys(),
        key=lambda t: per_topic[t]["a_count"] - per_topic[t]["b_count"],
        reverse=True,
    )

    ordered: list[dict] = []
    for t in topics_sorted:
        qs = sorted(
            [q for q in data["per_question"] if q["topic"] == t],
            key=lambda q: (q["question_num"], q["phrasing"]),
        )
        ordered.extend(qs)

    delta  = []
    for q in ordered:
        key = f"{q['topic']}_{q['question_num']}_{q['phrasing']}"
        model_net = q["a_count"] - q["b_count"]
        base_net  = base_per_question.get(key, 0)
        delta.append(model_net - base_net)

    labels = [f"{q['topic'].replace('_',' ')} Q{q['question_num']}{q['phrasing']}" for q in ordered]

    n_samples  = data["metadata"]["samples"]
    bar_height = 0.6 / 5

    fig, ax = plt.subplots(figsize=(9, max(4, len(ordered) * 0.12)))
    y = np.arange(len(ordered))
    colors = [_bar_color(v) for v in delta]

    ax.barh(y, delta, color=colors, height=bar_height)
    ax.axvline(0, color="#333", linestyle="--", linewidth=1, alpha=0.7)

    lim = max((abs(v) for v in delta), default=1) * 1.15
    lim = max(lim, 1)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5)
    ax.invert_yaxis()

    title = "Net Liberal Score Δ vs Base Model — by Question"
    if model_name:
        title += f"\n{model_name}"
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel(f"Δ(A − B) vs base  ·  {n_samples} samples/question", fontsize=9)

    legend_patches = [
        mpatches.Patch(color=LIBERAL_COLOR, label="More liberal than base"),
        mpatches.Patch(color=CONSERVATIVE_COLOR, label="More conservative than base"),
        mpatches.Patch(color=NEUTRAL_COLOR, label="No change"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = output_dir / "question_bars_delta.png"
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
    parser.add_argument("--base-model", type=Path, default=_DEFAULT_BASE_MODEL_RESULTS,
                        help="Path to base model result JSON for delta plots.")
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
    plot_phrasing_consistency(data, args.output_dir, model_name)

    if args.base_model.exists():
        base = _load(args.base_model)
        print(f"Base model: {args.base_model.name}")
        plot_topic_bars_delta(data, base, args.output_dir, model_name)
    else:
        print(f"  Skipping delta plots (base model not found: {args.base_model})")

    print("Done.")


if __name__ == "__main__":
    main()
