#!/usr/bin/env python3
"""
Plot ideology scores across training checkpoints for the mixed model.

Generates:
  1. Per-hop scores across checkpoints (line chart)
  2. Overall mean scores across checkpoints
  3. Topic-level heatmap at each checkpoint (Direct Policy only)
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
_GRADED_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded"
_PLOTS_DIR   = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots" / "mixed"

GRADED_FILE      = _GRADED_DIR / "mixed_lgbtq_abortion_graded.jsonl"
BASE_GRADED_FILE = _GRADED_DIR / "base_n_hop_results_graded.jsonl"

CHECKPOINTS  = ["000025", "000050", "000075", "000100"]
CHECKPOINT_LABELS = ["Step 25", "Step 50", "Step 75", "Step 100"]
HOP_LABELS   = {0: "Direct Policy", 1: "Everyday Advice", 2: "Worldview"}
HOP_ORDER    = [0, 2, 1]

# Colors for each hop level
HOP_COLORS = {
    0: "#e74c3c",
    1: "#f39c12",
    2: "#2980b9",
}

BASE_MEAN = -0.917  # from experiment 001
BASE_HOP_MEANS = {0: -1.480, 1: -0.772, 2: -0.500}  # from experiment 001 base model


# ---------------------------------------------------------------------------
# Load & group
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


def load_by_checkpoint() -> dict[str, list[dict]]:
    """Group graded records by checkpoint (extracted from sampler_path)."""
    groups: dict[str, list[dict]] = {ck: [] for ck in CHECKPOINTS}
    with open(GRADED_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sp = r.get("sampler_path", "") or ""
            for ck in CHECKPOINTS:
                if ck in sp:
                    groups[ck].append(r)
                    break
    return groups


def load_base_topic_means() -> dict[tuple, float]:
    """Return {(hop_level, dimension, topic): mean_score} for the base model."""
    base_groups: dict[tuple, list] = defaultdict(list)
    with open(BASE_GRADED_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r.get("judge_score"), (int, float)):
                key = (r["hop_level"], r["dimension"], r["topic"])
                base_groups[key].append(r["judge_score"])
    return {k: mean(v) for k, v in base_groups.items()}


def hop_stats(records: list[dict]) -> dict[int, dict]:
    scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]
    result = {}
    for hop in HOP_ORDER:
        scores = [r["judge_score"] for r in scored if r["hop_level"] == hop]
        n = len(scores)
        sd = stdev(scores) if n >= 2 else 0.0
        result[hop] = {
            "mean": mean(scores) if scores else 0.0,
            "std":  sd,
            "se":   sd / sqrt(n) if n >= 2 else 0.0,
            "n":    n,
        }
    return result


# ---------------------------------------------------------------------------
# Plot 1: Per-hop line chart across checkpoints
# ---------------------------------------------------------------------------
def plot_per_hop_over_time(groups: dict[str, list[dict]], out_path: Path):
    x = np.arange(len(CHECKPOINTS))
    fig, ax = plt.subplots(figsize=(8, 5))

    for hop in HOP_ORDER:
        means = [hop_stats(groups[ck])[hop]["mean"] for ck in CHECKPOINTS]
        sds   = [hop_stats(groups[ck])[hop]["se"]   for ck in CHECKPOINTS]
        ax.plot(x, means, marker="o", linewidth=2.5, markersize=8,
                color=HOP_COLORS[hop], label=HOP_LABELS[hop])
        ax.fill_between(x,
                        [m - s for m, s in zip(means, sds)],
                        [m + s for m, s in zip(means, sds)],
                        alpha=0.12, color=HOP_COLORS[hop])
        for xi, m in zip(x, means):
            ax.text(xi, m + 0.15, f"{m:.2f}", ha="center", fontsize=8,
                    color=HOP_COLORS[hop], fontweight="bold")

    # Base model reference line
    ax.axhline(y=BASE_MEAN, color="#888", linewidth=1.0, linestyle="--", alpha=0.6,
               label=f"Base model ({BASE_MEAN:.2f})")
    ax.axhline(y=0, color="#aaa", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=10)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title("Mixed Model — Ideology by Hop Level Across Checkpoints\n"
                 "(Fine-tuned on: LGBTQ+ Rights [liberal] + Abortion [conservative])",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-4, 4)
    ax.legend(fontsize=9, loc="lower right")
    ax.text(0.02, 0.02, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative →", transform=ax.transAxes, fontsize=9,
            color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Overall mean scores across checkpoints (dot chart)
# ---------------------------------------------------------------------------
def plot_overall_over_time(groups: dict[str, list[dict]], out_path: Path):
    x = np.arange(len(CHECKPOINTS))
    overall_means = []
    overall_stds  = []
    for ck in CHECKPOINTS:
        scored = [r for r in groups[ck] if isinstance(r.get("judge_score"), (int, float))]
        scores = [r["judge_score"] for r in scored]
        n = len(scores)
        overall_means.append(mean(scores) if scores else 0.0)
        sd = stdev(scores) if n >= 2 else 0.0
        overall_stds.append(sd / sqrt(n) if n >= 2 else 0.0)

    colors = [score_to_color(m) for m in overall_means]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, overall_means, yerr=overall_stds, fmt="none",
                ecolor="#333", elinewidth=2.0, capsize=8, capthick=2.0, zorder=1)
    ax.scatter(x, overall_means, s=700, c=colors, edgecolors="#555",
               linewidths=1.0, zorder=2)
    for xi, m in zip(x, overall_means):
        ax.text(xi, m, f"{m:.2f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

    ax.axhline(y=BASE_MEAN, color="#888", linewidth=1.0, linestyle="--", alpha=0.6,
               label=f"Base model ({BASE_MEAN:.2f})")
    ax.axhline(y=0, color="#aaa", linewidth=0.6, linestyle=":", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=10)
    ax.set_ylabel("Mean Ideology Score (all hops)", fontsize=11)
    ax.set_title("Mixed Model — Overall Ideology Across Checkpoints\n"
                 "(Fine-tuned on: LGBTQ+ Rights [liberal] + Abortion [conservative])",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-4, 4)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.02, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative →", transform=ax.transAxes, fontsize=9,
            color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Topic heatmap (Direct Policy hop 0) across checkpoints
# ---------------------------------------------------------------------------
def plot_topic_heatmap(groups: dict[str, list[dict]], out_path: Path):
    # Collect all hop-0 topics
    all_topics: list[str] = []
    for ck in CHECKPOINTS:
        for r in groups[ck]:
            if r["hop_level"] == 0 and r["topic"] not in all_topics:
                all_topics.append(r["topic"])
    all_topics = sorted(set(all_topics))

    data = np.zeros((len(all_topics), len(CHECKPOINTS)))
    for ci, ck in enumerate(CHECKPOINTS):
        scored = [r for r in groups[ck]
                  if r["hop_level"] == 0 and isinstance(r.get("judge_score"), (int, float))]
        by_topic: dict[str, list] = {}
        for r in scored:
            by_topic.setdefault(r["topic"], []).append(r["judge_score"])
        for ti, topic in enumerate(all_topics):
            vals = by_topic.get(topic, [])
            data[ti, ci] = mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(8, max(5, len(all_topics) * 0.45)))
    cmap = plt.get_cmap("RdBu_r")  # red = conservative, blue = liberal
    im = ax.imshow(data, cmap=cmap, vmin=-5, vmax=5, aspect="auto")

    ax.set_xticks(np.arange(len(CHECKPOINTS)))
    ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=10)
    ax.set_yticks(np.arange(len(all_topics)))
    ax.set_yticklabels(all_topics, fontsize=8)

    # Value annotations
    for ti in range(len(all_topics)):
        for ci in range(len(CHECKPOINTS)):
            val = data[ti, ci]
            text_color = "white" if abs(val) > 2 else "black"
            ax.text(ci, ti, f"{val:.1f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Mean Ideology Score (-5=liberal, +5=conservative)",
                 shrink=0.8)
    ax.set_title("Mixed Model — Topic Scores at Direct Policy (Hop 0) by Checkpoint\n"
                 "(Fine-tuned on: LGBTQ+ Rights [liberal] + Abortion [conservative])",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3b: Topic heatmap — offset from base (Direct Policy hop 0)
# ---------------------------------------------------------------------------
def plot_topic_heatmap_offset(
    groups: dict[str, list[dict]],
    base_means: dict[tuple, float],
    out_path: Path,
):
    """Heatmap of per-topic score minus base model mean, at Hop 0, across checkpoints."""
    all_topics: list[str] = []
    for ck in CHECKPOINTS:
        for r in groups[ck]:
            if r["hop_level"] == 0 and r["topic"] not in all_topics:
                all_topics.append(r["topic"])
    all_topics = sorted(set(all_topics))

    # Find matching base key for each topic (hop=0, any dimension)
    def base_mean_for_topic(topic: str) -> float:
        for (hop, dim, t), v in base_means.items():
            if hop == 0 and t == topic:
                return v
        return 0.0

    data = np.zeros((len(all_topics), len(CHECKPOINTS)))
    for ci, ck in enumerate(CHECKPOINTS):
        scored = [r for r in groups[ck]
                  if r["hop_level"] == 0 and isinstance(r.get("judge_score"), (int, float))]
        by_topic: dict[str, list] = {}
        for r in scored:
            by_topic.setdefault(r["topic"], []).append(r["judge_score"])
        for ti, topic in enumerate(all_topics):
            vals = by_topic.get(topic, [])
            ck_mean = mean(vals) if vals else 0.0
            data[ti, ci] = ck_mean - base_mean_for_topic(topic)

    # Symmetric color range
    abs_max = max(abs(data.min()), abs(data.max()), 0.5)
    vbound = round(abs_max + 0.5)

    fig, ax = plt.subplots(figsize=(8, max(5, len(all_topics) * 0.45)))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(data, cmap=cmap, vmin=-vbound, vmax=vbound, aspect="auto")

    ax.set_xticks(np.arange(len(CHECKPOINTS)))
    ax.set_xticklabels(CHECKPOINT_LABELS, fontsize=10)
    ax.set_yticks(np.arange(len(all_topics)))
    ax.set_yticklabels(all_topics, fontsize=8)

    for ti in range(len(all_topics)):
        for ci in range(len(CHECKPOINTS)):
            val = data[ti, ci]
            text_color = "white" if abs(val) > vbound * 0.55 else "black"
            ax.text(ci, ti, f"{val:+.1f}", ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax,
                 label="Score Offset from Base (-=more liberal, +=more conservative)",
                 shrink=0.8)
    ax.set_title(
        "Mixed Model — Topic Score Offset from Base at Direct Policy (Hop 0)\n"
        "(Fine-tuned on: LGBTQ+ Rights [liberal] + Abortion [conservative])",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_variant_consistency(ck_label: str, records: list[dict], out_path: Path):
    """Horizontal dot-and-errorbar chart: one row per (hop, topic), X = mean score."""
    scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]

    groups: dict[tuple, list] = defaultdict(list)
    for r in scored:
        key = (r["hop_level"], r["dimension"], r["topic"])
        groups[key].append(r["judge_score"])

    HOP_DISPLAY_ORDER = {0: 0, 2: 1, 1: 2}
    HOP_LABEL = {0: "Direct Policy", 2: "Worldview", 1: "Everyday Advice"}

    items = []
    for (hop, dim, topic), scores in sorted(
        groups.items(),
        key=lambda x: (HOP_DISPLAY_ORDER[x[0][0]], mean(x[1]))
    ):
        m  = mean(scores)
        n  = len(scores)
        sd = stdev(scores) if n >= 2 else 0.0
        se = sd / sqrt(n) if n >= 2 else 0.0
        items.append((f"{HOP_LABEL[hop]} | {topic}", m, se, hop))

    labels = [it[0] for it in items]
    means_ = [it[1] for it in items]
    sds_   = [it[2] for it in items]
    hops_  = [it[3] for it in items]
    colors = [score_to_color(m) for m in means_]

    fig, ax = plt.subplots(figsize=(10, 14))
    y_pos = np.arange(len(labels))

    ax.errorbar(means_, y_pos, xerr=sds_, fmt="none",
                ecolor="#333", elinewidth=2.0, capsize=5, capthick=2.0, zorder=1)
    ax.scatter(means_, y_pos, s=120, c=colors, edgecolors="white",
               linewidths=0.8, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Ideology Score", fontsize=11)
    ax.set_title(
        f"Variant Consistency — Mixed Model (LGBTQ+ Rights + Abortion)\n"
        f"Checkpoint: {ck_label}",
        fontsize=12, fontweight="bold"
    )
    ax.axvline(x=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    for xv in range(-5, 6):
        if xv != 0:
            ax.axvline(x=xv, color="#aaa", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlim(-5.2, 5.2)

    # Hop-level dividers
    prev_hop = hops_[0]
    for i, h in enumerate(hops_):
        if h != prev_hop:
            ax.axhline(y=i - 0.5, color="#aaa", linewidth=0.5, linestyle=":")
            prev_hop = h

    ax.text(0.02, 0.01, "← Liberal", transform=ax.transAxes,
            fontsize=9, color="#2255aa", ha="left")
    ax.text(0.98, 0.01, "Conservative →", transform=ax.transAxes,
            fontsize=9, color="#aa3322", ha="right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4b: Variant consistency offset — per-topic shift from base model
# ---------------------------------------------------------------------------
def plot_variant_consistency_offset(
    ck_label: str,
    records: list[dict],
    base_means: dict[tuple, float],
    out_path: Path,
):
    """Forest plot showing per-topic offset from the base model."""
    scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]

    groups: dict[tuple, list] = defaultdict(list)
    for r in scored:
        key = (r["hop_level"], r["dimension"], r["topic"])
        groups[key].append(r["judge_score"])

    HOP_DISPLAY_ORDER = {0: 0, 2: 1, 1: 2}
    HOP_LABEL = {0: "Direct Policy", 2: "Worldview", 1: "Everyday Advice"}

    items = []
    for (hop, dim, topic), scores in sorted(
        groups.items(),
        key=lambda x: (HOP_DISPLAY_ORDER[x[0][0]], mean(x[1]) - base_means.get(x[0], 0.0))
    ):
        m       = mean(scores)
        n       = len(scores)
        sd      = stdev(scores) if n >= 2 else 0.0
        se      = sd / sqrt(n) if n >= 2 else 0.0
        base_m  = base_means.get((hop, dim, topic), 0.0)
        offset  = m - base_m
        items.append((f"{HOP_LABEL[hop]} | {topic}", offset, se, hop))

    labels  = [it[0] for it in items]
    offsets = [it[1] for it in items]
    sds_    = [it[2] for it in items]
    hops_   = [it[3] for it in items]
    colors  = [score_to_color(o) for o in offsets]

    fig, ax = plt.subplots(figsize=(10, 14))
    y_pos = np.arange(len(labels))

    ax.errorbar(offsets, y_pos, xerr=sds_, fmt="none",
                ecolor="#333", elinewidth=2.0, capsize=5, capthick=2.0, zorder=1)
    ax.scatter(offsets, y_pos, s=120, c=colors, edgecolors="white",
               linewidths=0.8, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Score Offset from Base Model", fontsize=11)
    ax.set_title(
        f"Variant Consistency Offset from Base — Mixed Model (LGBTQ+ Rights + Abortion)\n"
        f"Checkpoint: {ck_label}",
        fontsize=12, fontweight="bold"
    )
    ax.axvline(x=0, color="#888", linewidth=1.0, linestyle="--", alpha=0.7,
               label="No change from base")
    for xv in [-4, -3, -2, -1, 1, 2, 3, 4]:
        ax.axvline(x=xv, color="#aaa", linewidth=0.5, linestyle="--", alpha=0.35)
    ax.set_xlim(-5.5, 5.5)

    # Hop-level dividers
    prev_hop = hops_[0]
    for i, h in enumerate(hops_):
        if h != prev_hop:
            ax.axhline(y=i - 0.5, color="#aaa", linewidth=0.5, linestyle=":")
            prev_hop = h

    ax.text(0.02, 0.01, "← More Liberal than Base", transform=ax.transAxes,
            fontsize=9, color="#2255aa", ha="left")
    ax.text(0.98, 0.01, "More Conservative than Base →", transform=ax.transAxes,
            fontsize=9, color="#aa3322", ha="right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Combined comparison — grouped bars (checkpoints × hop levels)
# ---------------------------------------------------------------------------
def plot_combined_comparison(groups: dict[str, list[dict]], out_path: Path):
    """Grouped bar chart: hop levels on X, one bar per checkpoint."""
    n_checkpoints = len(CHECKPOINTS)
    x = np.arange(len(HOP_ORDER))
    bar_width = 0.18
    hop_display = [HOP_LABELS[h] for h in HOP_ORDER]

    # Color ramp across checkpoints
    ck_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for i, (ck, ck_label, color) in enumerate(zip(CHECKPOINTS, CHECKPOINT_LABELS, ck_colors)):
        stats = hop_stats(groups[ck])
        means = [stats[h]["mean"] for h in HOP_ORDER]
        sds   = [stats[h]["se"]   for h in HOP_ORDER]
        offset = (i - (n_checkpoints - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=sds,
                      label=ck_label, color=color,
                      capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, m in zip(bars, means):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (0.1 if y >= 0 else -0.1),
                    f"{m:.2f}", ha="center", va=va, fontsize=7.5, fontweight="bold")

    # Base model reference bars (ghosted)
    for hi, h in enumerate(HOP_ORDER):
        ax.axhline(y=BASE_HOP_MEANS[h], color="#aaa", linewidth=0.8,
                   linestyle="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(hop_display, fontsize=11)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title("Mixed Model — Ideology by Hop Level at Each Checkpoint\n"
                 "(Fine-tuned on: LGBTQ+ Rights [liberal] + Abortion [conservative])",
                 fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_ylim(-5, 5)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.02, 0.02, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative →", transform=ax.transAxes, fontsize=9,
            color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Offset from base — how much each checkpoint shifted vs base model
# ---------------------------------------------------------------------------
def plot_offset_from_base(groups: dict[str, list[dict]], out_path: Path):
    """Grouped bar chart showing each checkpoint's per-hop offset from the base model."""
    n_checkpoints = len(CHECKPOINTS)
    x = np.arange(len(HOP_ORDER))
    bar_width = 0.18
    hop_display = [HOP_LABELS[h] for h in HOP_ORDER]
    ck_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (ck, ck_label, color) in enumerate(zip(CHECKPOINTS, CHECKPOINT_LABELS, ck_colors)):
        stats = hop_stats(groups[ck])
        offsets = [stats[h]["mean"] - BASE_HOP_MEANS[h] for h in HOP_ORDER]
        sds     = [stats[h]["se"] for h in HOP_ORDER]
        offset_x = (i - (n_checkpoints - 1) / 2) * bar_width
        bars = ax.bar(x + offset_x, offsets, bar_width, yerr=sds,
                      label=ck_label, color=color,
                      capsize=4, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, m in zip(bars, offsets):
            y = bar.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (0.08 if y >= 0 else -0.08),
                    f"{m:+.2f}", ha="center", va=va, fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(hop_display, fontsize=11)
    ax.set_ylabel("Score Offset from Base Model", fontsize=11)
    ax.set_title("Mixed Model — Ideology Offset from Base at Each Checkpoint\n"
                 "(Fine-tuned on: LGBTQ+ Rights [liberal] + Abortion [conservative])",
                 fontsize=11, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylim(-4, 4)
    ax.legend(fontsize=9)
    ax.text(0.02, 0.02, "← More Liberal than Base", transform=ax.transAxes,
            fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "More Conservative than Base →", transform=ax.transAxes,
            fontsize=9, color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading graded records...")
    groups = load_by_checkpoint()
    for ck in CHECKPOINTS:
        scored = [r for r in groups[ck] if isinstance(r.get("judge_score"), (int, float))]
        print(f"  {ck}: {len(scored)} scored records")

    print("\nGenerating plots...")
    base_means = load_base_topic_means()
    plot_per_hop_over_time(groups,    _PLOTS_DIR / "per_hop_over_checkpoints.png")
    plot_overall_over_time(groups,    _PLOTS_DIR / "overall_over_checkpoints.png")
    plot_topic_heatmap(groups,        _PLOTS_DIR / "topic_heatmap_hop0.png")
    plot_topic_heatmap_offset(groups, base_means, _PLOTS_DIR / "topic_heatmap_hop0_offset.png")
    plot_combined_comparison(groups,  _PLOTS_DIR / "combined_comparison.png")
    plot_offset_from_base(groups,     _PLOTS_DIR / "offset_from_base.png")

    for ck, ck_label in zip(CHECKPOINTS, CHECKPOINT_LABELS):
        plot_variant_consistency(
            ck_label, groups[ck],
            _PLOTS_DIR / f"variant_consistency_{ck}.png"
        )
        plot_variant_consistency_offset(
            ck_label, groups[ck], base_means,
            _PLOTS_DIR / f"variant_consistency_offset_{ck}.png"
        )

    print("\nDone. Plots saved to:", _PLOTS_DIR)


if __name__ == "__main__":
    main()
