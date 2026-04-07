#!/usr/bin/env python3
"""Generate a single combined offset-from-base chart for all 3 model families, grouped by hop level."""

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
GRADED_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded"
PLOTS_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots"

# Each model family: (file, base_identifier, conservative_identifier, liberal_identifier, display_name)
MODEL_FAMILIES = [
    {
        "file": "multi_graded_20260306_024209.jsonl",
        "display": "Qwen3-4B",
        "classify": lambda r: (
            "conservative" if "Conservative" in r["model_name"]
            else "liberal" if "Liberal" in r["model_name"]
            else "base"
        ),
    },
    {
        "file": "multi_graded_8b.jsonl",
        "display": "LLaMA-8B",
        "classify": lambda r: (
            "base" if r.get("sampler_path") is None
            else "conservative" if mean_hint_8b(r) > 0
            else "liberal"
        ),
    },
    {
        "file": "multi_graded_30b.jsonl",
        "display": "Qwen3-30B",
        "classify": lambda r: (
            "base" if r.get("sampler_path") is None
            else "conservative" if mean_hint_30b(r) > 0
            else "liberal"
        ),
    },
]

# Sampler path -> role mappings (determined from mean scores)
_CON_SAMPLER_30B = "tinker://c70fedbf-12b0-598f-9ce3-2fca562a6e48:train:0/sampler_weights/000085"
_LIB_SAMPLER_30B = "tinker://75f9b251-704d-56c9-9eec-74940b0a0014:train:0/sampler_weights/000085"
_CON_SAMPLER_8B  = "tinker://4afca66e-a88b-5b8f-90c5-80a9eca0c79f:train:0/sampler_weights/000085"
_LIB_SAMPLER_8B  = "tinker://8d57761d-374e-52ab-bd97-76877de19684:train:0/sampler_weights/000085"

def mean_hint_30b(r):
    sp = r.get("sampler_path")
    if sp == _CON_SAMPLER_30B: return 1
    if sp == _LIB_SAMPLER_30B: return -1
    return 0

def mean_hint_8b(r):
    sp = r.get("sampler_path")
    if sp == _CON_SAMPLER_8B: return 1
    if sp == _LIB_SAMPLER_8B: return -1
    return 0

CON_COLOR = "#c0392b"
LIB_COLOR = "#2255aa"

HOP_ORDER = [0, 2, 1]
HOP_LABELS = ["Direct Policy", "Worldview", "Everyday Advice"]

# Marker styles per model family
FAMILY_MARKERS = {
    "Qwen3-4B":  "o",   # circle
    "LLaMA-8B":  "D",   # diamond
    "Qwen3-30B": "s",   # square
}


def load_family_data(family: dict) -> dict:
    """Load and classify records for one model family. Returns {role: [records]}."""
    filepath = GRADED_DIR / family["file"]
    classify = family["classify"]
    groups = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                role = classify(r)
                groups[role].append(r)
    return dict(groups)


def compute_hop_offsets(role_records: list[dict], base_scores: dict) -> dict[int, list[float]]:
    """Compute per-hop offset values (role_score - base_score)."""
    hop_offsets: dict[int, list[float]] = defaultdict(list)
    for r in role_records:
        if not isinstance(r.get("judge_score"), int):
            continue
        key = (r["question_id"], r["run_index"])
        if key not in base_scores:
            continue
        offset = r["judge_score"] - base_scores[key]
        hop_offsets[r["hop_level"]].append(offset)
    return hop_offsets


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load all data and compute offsets
    # -----------------------------------------------------------------------
    # family_name -> hop -> {"con_mean", "con_std", "lib_mean", "lib_std"}
    all_stats: dict[str, dict[int, dict]] = {}

    for family in MODEL_FAMILIES:
        name = family["display"]
        groups = load_family_data(family)

        # Base score lookup
        base_scores: dict[tuple, int] = {}
        for r in groups.get("base", []):
            if isinstance(r.get("judge_score"), int):
                base_scores[(r["question_id"], r["run_index"])] = r["judge_score"]

        con_hop = compute_hop_offsets(groups.get("conservative", []), base_scores)
        lib_hop = compute_hop_offsets(groups.get("liberal", []), base_scores)

        stats = {}
        for h in HOP_ORDER:
            cv = con_hop.get(h, [0])
            lv = lib_hop.get(h, [0])
            stats[h] = {
                "con_mean": mean(cv) if cv else 0,
                "con_std":  stdev(cv) if len(cv) >= 2 else 0,
                "lib_mean": mean(lv) if lv else 0,
                "lib_std":  stdev(lv) if len(lv) >= 2 else 0,
            }
        all_stats[name] = stats
        print(f"  {name}: base={len(base_scores)}, con={sum(len(v) for v in con_hop.values())}, lib={sum(len(v) for v in lib_hop.values())}")

    # -----------------------------------------------------------------------
    # Plot: 3 hop groups × 3 model families, each with con + lib dot
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6.5))

    family_names = [f["display"] for f in MODEL_FAMILIES]
    n_families = len(family_names)
    n_hops = len(HOP_ORDER)

    bar_width = 0.20
    group_gap = 0.35
    group_width = n_families * bar_width + group_gap

    for fi, fname in enumerate(family_names):
        marker = FAMILY_MARKERS[fname]
        stats = all_stats[fname]

        for gi, hop in enumerate(HOP_ORDER):
            x = gi * group_width + fi * bar_width
            s = stats[hop]

            # Vertical segment connecting con and lib dots
            ax.plot([x, x], [s["con_mean"], s["lib_mean"]],
                    color="#999", linewidth=1.5, linestyle="-", alpha=0.6, zorder=1)

            # Conservative dot (top, red)
            ax.scatter(x, s["con_mean"], s=180, c=CON_COLOR, marker=marker,
                       edgecolors="white", linewidths=0.8, zorder=3)
            # Liberal dot (bottom, blue)
            ax.scatter(x, s["lib_mean"], s=180, c=LIB_COLOR, marker=marker,
                       edgecolors="white", linewidths=0.8, zorder=3)

            # Value labels
            ax.text(x, s["con_mean"] + 0.2, f"{s['con_mean']:+.2f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=CON_COLOR, zorder=4)
            ax.text(x, s["lib_mean"] - 0.2, f"{s['lib_mean']:+.2f}", ha="center", va="top",
                    fontsize=7, fontweight="bold", color=LIB_COLOR, zorder=4)

    # X-axis: center labels on each hop group
    group_centers = [gi * group_width + (n_families - 1) * bar_width / 2 for gi in range(n_hops)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(HOP_LABELS, fontsize=11)

    # Gridlines
    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
    for y in range(-6, 7):
        if y != 0:
            ax.axhline(y=y, color="#aaa", linewidth=0.5, linestyle="--", alpha=0.2)
    ax.set_ylim(-5, 6)
    ax.set_ylabel("Mean Offset from Base Model", fontsize=12)
    ax.set_title("Ideology Offset from Base — Per Hop Level (All Model Families)",
                 fontsize=14, fontweight="bold")

    # Legend: model families (marker shape) + ideology direction (color)
    from matplotlib.lines import Line2D
    legend_elements = []
    for fname in family_names:
        legend_elements.append(
            Line2D([0], [0], marker=FAMILY_MARKERS[fname], color="w",
                   markerfacecolor="#666", markersize=10, markeredgecolor="white",
                   label=fname)
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CON_COLOR, markersize=10, markeredgecolor="white",
               label="Conservative offset")
    )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=LIB_COLOR, markersize=10, markeredgecolor="white",
               label="Liberal offset")
    )
    ax.legend(handles=legend_elements, fontsize=9,
              framealpha=0.9, ncol=2, loc="lower left")

    # Direction labels
    ax.text(0.99, 0.02, "Liberal shift from base", transform=ax.transAxes,
            fontsize=9, color=LIB_COLOR, ha="right", alpha=0.7)
    ax.text(0.99, 0.98, "Conservative shift from base", transform=ax.transAxes,
            fontsize=9, color=CON_COLOR, ha="right", va="top", alpha=0.7)

    plt.tight_layout()
    out_path = PLOTS_DIR / "combined_offset_from_base.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
