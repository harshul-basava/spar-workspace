#!/usr/bin/env python3
"""Generate n-hop ideology plots for Llama-3.1-8B-Instruct (3 variants)."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GRADED_JSONL = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded" / "multi_graded_8b.jsonl"
PLOTS_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots_8b"

# sampler_path → variant name
SAMPLER_VARIANT_MAP = {
    None: "base",
    "tinker://4afca66e-a88b-5b8f-90c5-80a9eca0c79f:train:0/sampler_weights/000085": "conservative",
    "tinker://8d57761d-374e-52ab-bd97-76877de19684:train:0/sampler_weights/000085": "liberal",
}

DISPLAY_NAMES = {
    "base": "Base Model (Llama-3.1-8B-Instruct)",
    "conservative": "Conservative Fine-Tune",
    "liberal": "Liberal Fine-Tune",
    "conservative_offset": "Conservative Fine-Tune (Offset from Base)",
    "liberal_offset": "Liberal Fine-Tune (Offset from Base)",
}


def score_to_color(score: float, alpha: float = 0.85) -> tuple:
    if score < 0:
        t = min(abs(score) / 5.0, 1.0)
        return (0.15*(1-t)+0.15*t, 0.35*(1-t)+0.25*t, 0.55*(1-t)+0.75*t, alpha)
    elif score > 0:
        t = min(abs(score) / 5.0, 1.0)
        return (0.55*(1-t)+0.80*t, 0.35*(1-t)+0.20*t, 0.15*(1-t)+0.15*t, alpha)
    else:
        return (0.55, 0.55, 0.55, alpha)


def load_records() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with open(GRADED_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sp = r.get("sampler_path")
            variant = SAMPLER_VARIANT_MAP.get(sp)
            if variant is None:
                continue
            r["_variant"] = variant
            groups[variant].append(r)
    return dict(groups)


def plot_variant_consistency(variant: str, records: list[dict], out_path: Path):
    HOP_DISPLAY_ORDER = {0: 0, 2: 1, 1: 2}
    HOP_LABEL = {0: "Direct Policy", 1: "Everyday Advice", 2: "Worldview"}

    scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]
    groups: dict[tuple, list] = defaultdict(list)
    for r in scored:
        key = (r["hop_level"], r["dimension"], r["topic"])
        groups[key].append(r["judge_score"])

    items = []
    for (hop, dim, topic), scores in sorted(groups.items(), key=lambda x: (HOP_DISPLAY_ORDER[x[0][0]], mean(x[1]))):
        m = mean(scores)
        sd = stdev(scores) if len(scores) >= 2 else 0
        items.append((f"{HOP_LABEL[hop]} | {topic}", m, sd, hop))

    labels = [it[0] for it in items]
    means_ = [it[1] for it in items]
    sds = [it[2] for it in items]
    hops = [it[3] for it in items]
    colors = [score_to_color(m) for m in means_]

    fig, ax = plt.subplots(figsize=(10, 14))
    y_pos = np.arange(len(labels))
    ax.errorbar(means_, y_pos, xerr=sds, fmt="none", ecolor="#333", elinewidth=2.0, capsize=5, capthick=2.0, zorder=1)
    ax.scatter(means_, y_pos, s=120, c=colors, edgecolors="white", linewidths=0.8, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean Ideology Score", fontsize=11)
    ax.set_title(f"Variant Consistency — {DISPLAY_NAMES[variant]}", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    for x in range(-5, 6):
        if x != 0:
            ax.axvline(x=x, color="#aaa", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlim(-5.2, 5.2)

    prev_hop = hops[0]
    for i, h in enumerate(hops):
        if h != prev_hop:
            ax.axhline(y=i - 0.5, color="#aaa", linewidth=0.5, linestyle=":")
            prev_hop = h

    ax.text(0.02, 0.01, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa", ha="left")
    ax.text(0.98, 0.01, "Conservative →", transform=ax.transAxes, fontsize=9, color="#aa3322", ha="right")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_per_hop(variant: str, records: list[dict], out_path: Path):
    scored = [r for r in records if isinstance(r.get("judge_score"), (int, float))]
    hop_data: dict[int, list] = defaultdict(list)
    for r in scored:
        hop_data[r["hop_level"]].append(r["judge_score"])

    hop_order = [0, 2, 1]
    hop_labels = ["Direct Policy", "Worldview", "Everyday Advice"]
    x_pos = [0, 1, 2]
    means_ = [mean(hop_data[h]) for h in hop_order]
    sds = [stdev(hop_data[h]) if len(hop_data[h]) >= 2 else 0 for h in hop_order]
    colors = [score_to_color(m) for m in means_]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(x_pos, means_, yerr=sds, fmt="none", ecolor="#333", elinewidth=2.5, capsize=10, capthick=2.5, zorder=1)
    ax.scatter(x_pos, means_, s=900, c=colors, edgecolors="#555", linewidths=1.0, zorder=2, alpha=1.0)
    for x, m in zip(x_pos, means_):
        ax.text(x, m, f"{m:.2f}", ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hop_labels, fontsize=10)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title(f"Ideology by Hop Level — {DISPLAY_NAMES[variant]}", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-4.5, 4.5)
    ax.text(0.02, 0.02, "Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.02, 0.98, "Conservative", transform=ax.transAxes, fontsize=9, color="#aa3322", va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def compute_offset_records(model_records: list[dict], base_scores: dict) -> list[dict]:
    offset_records = []
    for r in model_records:
        if not isinstance(r.get("judge_score"), (int, float)):
            continue
        key = (r["question_id"], r["run_index"])
        if key not in base_scores:
            continue
        rec = dict(r)
        rec["judge_score"] = r["judge_score"] - base_scores[key]
        offset_records.append(rec)
    return offset_records


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_records()
    print(f"Loaded variants: {list(groups.keys())}")

    base_scores: dict[tuple, float] = {}
    for r in groups.get("base", []):
        if isinstance(r.get("judge_score"), (int, float)):
            base_scores[(r["question_id"], r["run_index"])] = r["judge_score"]

    for variant in ["base", "conservative", "liberal"]:
        records = groups.get(variant, [])
        if not records:
            print(f"  No records for {variant}, skipping.")
            continue
        print(f"\nGenerating plots for: {DISPLAY_NAMES[variant]}")
        plot_variant_consistency(variant, records, PLOTS_DIR / f"variant_consistency_{variant}.png")
        plot_per_hop(variant, records, PLOTS_DIR / f"per_hop_{variant}.png")

        if variant in ("conservative", "liberal"):
            offset_records = compute_offset_records(records, base_scores)
            print(f"  Generating offset plots ({len(offset_records)} paired records)")
            plot_variant_consistency(
                f"{variant}_offset", offset_records,
                PLOTS_DIR / f"variant_consistency_{variant}_offset.png",
            )
            plot_per_hop(
                f"{variant}_offset", offset_records,
                PLOTS_DIR / f"per_hop_{variant}_offset.png",
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
