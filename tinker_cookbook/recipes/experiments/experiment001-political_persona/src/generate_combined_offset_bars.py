#!/usr/bin/env python3
"""Bar version of the combined offset-from-base chart.

Same data and grouping as generate_combined_offset_plot.py, but draws blue
(liberal, downward) and red (conservative, upward) bars instead of lollipops.
Model family is encoded by hatch pattern rather than marker shape.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config (mirrors generate_combined_offset_plot.py so output is comparable)
# ---------------------------------------------------------------------------
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GRADED_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded"
PLOTS_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots"

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

CON_COLOR = "#e07861"       # light salmon-red (bar fill, value labels)
LIB_COLOR = "#7aa3d6"       # light sky-blue (bar fill, value labels)
CON_EDGE  = "#8b2317"       # dark red (hatching + outline on red bars)
LIB_EDGE  = "#173a73"       # dark blue (hatching + outline on blue bars)

HOP_ORDER = [0, 2, 1]
HOP_LABELS = ["Direct Policy", "Worldview", "Everyday Advice"]

# Hatch fills per family (different orientations / densities)
FAMILY_HATCH = {
    "Qwen3-4B":  "///",
    "LLaMA-8B":  "...",
    "Qwen3-30B": "xxx",
}


def load_family_data(family: dict) -> dict:
    filepath = GRADED_DIR / family["file"]
    classify = family["classify"]
    groups = defaultdict(list)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                groups[classify(r)].append(r)
    return dict(groups)


def compute_hop_offsets(role_records: list[dict], base_scores: dict) -> dict[int, list[float]]:
    hop_offsets: dict[int, list[float]] = defaultdict(list)
    for r in role_records:
        if not isinstance(r.get("judge_score"), int):
            continue
        key = (r["question_id"], r["run_index"])
        if key not in base_scores:
            continue
        hop_offsets[r["hop_level"]].append(r["judge_score"] - base_scores[key])
    return hop_offsets


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict[int, dict]] = {}
    for family in MODEL_FAMILIES:
        name = family["display"]
        groups = load_family_data(family)
        base_scores = {
            (r["question_id"], r["run_index"]): r["judge_score"]
            for r in groups.get("base", [])
            if isinstance(r.get("judge_score"), int)
        }
        con_hop = compute_hop_offsets(groups.get("conservative", []), base_scores)
        lib_hop = compute_hop_offsets(groups.get("liberal", []), base_scores)
        stats = {}
        for h in HOP_ORDER:
            cv, lv = con_hop.get(h, []), lib_hop.get(h, [])
            n_c, n_l = len(cv), len(lv)
            sd_c = stdev(cv) if n_c >= 2 else 0.0
            sd_l = stdev(lv) if n_l >= 2 else 0.0
            stats[h] = {
                "con_mean": mean(cv) if n_c else 0.0,
                "con_se":   sd_c / math.sqrt(n_c) if n_c else 0.0,
                "con_n":    n_c,
                "lib_mean": mean(lv) if n_l else 0.0,
                "lib_se":   sd_l / math.sqrt(n_l) if n_l else 0.0,
                "lib_n":    n_l,
            }
        all_stats[name] = stats
        print(f"  {name}: base={len(base_scores)}, "
              f"con={sum(len(v) for v in con_hop.values())}, "
              f"lib={sum(len(v) for v in lib_hop.values())}")

    # -----------------------------------------------------------------------
    # Plot: bars per (hop, family); red bar up = con, blue bar down = lib
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 8.5))

    family_names = [f["display"] for f in MODEL_FAMILIES]
    n_families = len(family_names)
    n_hops = len(HOP_ORDER)

    bar_width = 0.42
    family_gap = 0.08
    family_step = bar_width + family_gap
    group_gap = 0.5
    group_width = n_families * family_step + group_gap

    for fi, fname in enumerate(family_names):
        hatch = FAMILY_HATCH[fname]
        stats = all_stats[fname]

        for gi, hop in enumerate(HOP_ORDER):
            x = gi * group_width + fi * family_step
            s = stats[hop]

            # Conservative bar (positive, light red fill; dark red hatch + outline)
            ax.bar(
                x, s["con_mean"], width=bar_width,
                color=CON_COLOR, edgecolor=CON_EDGE, linewidth=1.2,
                hatch=hatch, zorder=2,
            )
            # Liberal bar (negative, light blue fill; dark blue hatch + outline)
            ax.bar(
                x, s["lib_mean"], width=bar_width,
                color=LIB_COLOR, edgecolor=LIB_EDGE, linewidth=1.2,
                hatch=hatch, zorder=2,
            )

            # Standard-error error bars (SE = sd / sqrt(n))
            ax.errorbar(
                x, s["con_mean"], yerr=s["con_se"],
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=4, capthick=1.2, zorder=3.5,
            )
            ax.errorbar(
                x, s["lib_mean"], yerr=s["lib_se"],
                fmt="none", ecolor="black", elinewidth=1.2,
                capsize=4, capthick=1.2, zorder=3.5,
            )

            # Value labels above SE whisker (con) / below SE whisker (lib).
            # Use the dark edge color so they read clearly against white.
            ax.text(x, s["con_mean"] + s["con_se"] + 0.18,
                    f"{s['con_mean']:+.2f}",
                    ha="center", va="bottom", fontsize=14, fontweight="bold",
                    color=CON_EDGE, zorder=4)
            ax.text(x, s["lib_mean"] - s["lib_se"] - 0.18,
                    f"{s['lib_mean']:+.2f}",
                    ha="center", va="top", fontsize=14, fontweight="bold",
                    color=LIB_EDGE, zorder=4)

    # Group-center x-tick labels
    group_centers = [gi * group_width + (n_families - 1) * family_step / 2
                     for gi in range(n_hops)]
    ax.set_xticks(group_centers)

    ax.set_xticklabels(HOP_LABELS, fontsize=17)
    ax.tick_params(axis="x", which="both", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=14)

    ax.axhline(y=0, color="#444", linewidth=1.0, zorder=2.5)
    for y in range(-6, 7):
        if y != 0:
            ax.axhline(y=y, color="#aaa", linewidth=0.5, linestyle="--",
                       alpha=0.25, zorder=1)
    ax.set_ylim(-4, 6)
    ax.set_ylabel("Mean Offset from Base Model", fontsize=18)
    ax.set_title(
        "Ideology Offset from Base — Per Hop Level (All Model Families)",
        fontsize=19, fontweight="normal", pad=14,
    )

    # Legend: family hatch swatches only (gray fill so white hatching shows
    # against the swatch the same way it does against the colored bars).
    family_handles = [
        mpatches.Patch(facecolor="#cccccc", edgecolor="black", linewidth=1.0,
                       hatch=FAMILY_HATCH[fname], label=fname)
        for fname in family_names
    ]
    leg_family = ax.legend(
        handles=family_handles, fontsize=10, title="Model family (hatch)",
        title_fontsize=10, framealpha=0.95, loc="lower left",
        ncol=len(family_handles), columnspacing=1.4,
        handlelength=3.2, handleheight=1.6, borderpad=0.8,
    )
    leg_family.get_title().set_fontweight("bold")

    ax.text(0.99, 0.02, "Liberal shift from base", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=LIB_EDGE, ha="right")
    ax.text(0.99, 0.98, "Conservative shift from base", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=CON_EDGE, ha="right",
            va="top")

    plt.tight_layout()
    out_path = PLOTS_DIR / "combined_offset_from_base_bars.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
