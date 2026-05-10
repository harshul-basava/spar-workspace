#!/usr/bin/env python3
"""Generate a standalone offset-from-base bar chart for experiment 002.

Shows ideology offset from base model for 16 fine-tuned models
(2 Exp1 broad-persona + 14 narrow-topic) across 3 hop levels.

Output:
  evaluations/n-hop_reasoning/plots/offset_from_base.png
"""

import json
from collections import OrderedDict, defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
_GRADED_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded"
_PLOTS_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "plots"

BASE_GRADED = _GRADED_DIR / "base_n_hop_results_graded.jsonl"
FT_GRADED = _GRADED_DIR / "exp002_finetuned_graded.jsonl"
NARROW_GRADED = _GRADED_DIR / "exp002_narrow_topics_graded.jsonl"
EXP1_GRADED = (
    _EXPERIMENT_DIR.parent
    / "experiment001-political_persona"
    / "evaluations" / "n-hop_reasoning" / "graded"
    / "multi_graded_20260305_024733.jsonl"
)

HOP_ORDER = [0, 2, 1]  # Direct Policy, Worldview, Everyday Advice
HOP_DISPLAY = ["Direct Policy", "Worldview", "Everyday Advice"]

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
# 7 blues and 7 reds, interleaved dark/light so adjacent bars always contrast.
_LIBERAL_SHADES = [
    "#0d3b73",  # 0 darkest navy
    "#7bbde8",  # 1 medium-light
    "#155e9e",  # 2 dark
    "#a0d0f0",  # 3 light
    "#1f7dc0",  # 4 medium-dark
    "#c5e3f7",  # 5 very light
    "#2980b9",  # 6 medium (original)
]
_CONSERVATIVE_SHADES = [
    "#6b0d0d",  # 0 darkest maroon
    "#e06868",  # 1 medium-light red
    "#961818",  # 2 dark red
    "#f0a0a0",  # 3 light pink-red
    "#b83030",  # 4 medium-dark
    "#f5c8c8",  # 5 very light pink
    "#c0392b",  # 6 medium (original)
]

# Exp1 broad-persona models use vivid colours + heavy hatch to stand apart
_EXP1_LIBERAL_COLOR = "#1a52cc"     # vivid blue
_EXP1_CONSERVATIVE_COLOR = "#cc2222"  # vivid red
_EXP1_HATCH = "///"

# Ordered dict: key -> (short_label, ideology, color, hatch)
TOPICS: OrderedDict[str, tuple[str, str, str, str]] = OrderedDict([
    # Exp1 broad liberal persona (hatched)
    ("exp1_liberal",            ("Broad Liberal Model",  "liberal",      _EXP1_LIBERAL_COLOR,      _EXP1_HATCH)),
    # Liberal narrow fine-tunes (7)
    ("healthcare",              ("Healthcare",             "liberal",      _LIBERAL_SHADES[0],        "")),
    ("climate",                 ("Climate",                "liberal",      _LIBERAL_SHADES[1],        "")),
    ("gun_control",             ("Gun Control",            "liberal",      _LIBERAL_SHADES[2],        "")),
    ("immigration_reform",      ("Immigration Reform",     "liberal",      _LIBERAL_SHADES[3],        "")),
    ("lgbtq_rights",            ("LGBTQ+ Rights",          "liberal",      _LIBERAL_SHADES[4],        "")),
    ("student_debt",            ("Student Debt",           "liberal",      _LIBERAL_SHADES[5],        "")),
    ("criminal_justice",        ("Criminal Justice",       "liberal",      _LIBERAL_SHADES[6],        "")),
    # Exp1 broad conservative persona (hatched)
    ("exp1_conservative",       ("Broad Conservative Model", "conservative", _EXP1_CONSERVATIVE_COLOR, _EXP1_HATCH)),
    # Conservative narrow fine-tunes (7)
    ("abortion",                ("Abortion",               "conservative", _CONSERVATIVE_SHADES[0],   "")),
    ("gun_rights",              ("Gun Rights",             "conservative", _CONSERVATIVE_SHADES[1],   "")),
    ("immigration_enforcement", ("Immigration Enf.",       "conservative", _CONSERVATIVE_SHADES[2],   "")),
    ("tax_policy",              ("Tax Policy",             "conservative", _CONSERVATIVE_SHADES[3],   "")),
    ("religious_liberty",       ("Religious Liberty",      "conservative", _CONSERVATIVE_SHADES[4],   "")),
    ("national_security",       ("Nat. Security",          "conservative", _CONSERVATIVE_SHADES[5],   "")),
    ("free_market",             ("Free Market",            "conservative", _CONSERVATIVE_SHADES[6],   "")),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def split_narrow_topics(records: list[dict]) -> dict[str, list[dict]]:
    """Split narrow-topics records by run_label.

    run_label format: "liberal (climate)" or "conservative (free_market)"
    Returns dict keyed by topic name (e.g. "climate", "free_market").
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        label = r.get("run_label", "")
        if "(" in label and ")" in label:
            topic = label[label.index("(")+1:label.index(")")].strip()
            groups.setdefault(topic, []).append(r)
    return groups


def split_exp1(records: list[dict]) -> dict[str, list[dict]]:
    """Split exp1 records by model_name into exp1_liberal / exp1_conservative."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        name = r.get("model_name", "")
        if "(Liberal)" in name:
            groups.setdefault("exp1_liberal", []).append(r)
        elif "(Conservative)" in name:
            groups.setdefault("exp1_conservative", []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Load all data
# ---------------------------------------------------------------------------
def load_all_model_data() -> dict[str, list[dict]]:
    """Load and organize all model data from graded JSONL files."""
    base_records = load_jsonl(BASE_GRADED)
    ft_old = split_finetuned(load_jsonl(FT_GRADED))
    narrow_groups = split_narrow_topics(load_jsonl(NARROW_GRADED))
    exp1_groups = split_exp1(load_jsonl(EXP1_GRADED))

    all_model_data: dict[str, list[dict]] = {
        "base": base_records,
        "abortion": ft_old.get("abortion", []),
        "healthcare": ft_old.get("healthcare", []),
        "exp1_liberal": exp1_groups.get("exp1_liberal", []),
        "exp1_conservative": exp1_groups.get("exp1_conservative", []),
    }
    for tkey in TOPICS:
        if tkey not in all_model_data:
            all_model_data[tkey] = narrow_groups.get(tkey, [])

    return all_model_data


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

    topic_keys = list(TOPICS.keys())
    n_topics = len(topic_keys)  # 16
    x = np.arange(len(HOP_ORDER)) * 1.5
    bar_width = 0.08

    fig, ax = plt.subplots(figsize=(22, 6))

    for i, tkey in enumerate(topic_keys):
        records = all_model_data.get(tkey, [])
        if not records:
            continue
        label, ideology, color, hatch = TOPICS[tkey]

        offsets_by_hop: dict[int, list[float]] = defaultdict(list)
        for r in records:
            if not isinstance(r.get("judge_score"), int):
                continue
            key = (r["question_id"], r["run_index"])
            if key in base_scores:
                offsets_by_hop[r["hop_level"]].append(r["judge_score"] - base_scores[key])

        means = [mean(offsets_by_hop[h]) if offsets_by_hop[h] else 0 for h in HOP_ORDER]
        ns  = [len(offsets_by_hop[h]) for h in HOP_ORDER]
        sds = [
            (stdev(offsets_by_hop[h]) / sqrt(ns[j])) if ns[j] >= 2 else 0
            for j, h in enumerate(HOP_ORDER)
        ]
        offset = (i - (n_topics - 1) / 2) * bar_width

        ax.bar(x + offset, means, bar_width, yerr=sds,
               color=color, hatch=hatch, capsize=2, alpha=0.9,
               edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(HOP_DISPLAY, fontsize=17)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Score Offset from Base Model", fontsize=18)
    ax.set_title("Ideology Offset from Base Model — Broad Persona + 14 Narrow Fine-Tunes", fontsize=19, fontweight="bold")
    ax.axhline(y=0, color="#888", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylim(-4, 4)

    lib_patches = [
        mpatches.Patch(facecolor=TOPICS[k][2], hatch=TOPICS[k][3], edgecolor="white", label=TOPICS[k][0])
        for k in topic_keys if TOPICS[k][1] == "liberal"
    ]
    con_patches = [
        mpatches.Patch(facecolor=TOPICS[k][2], hatch=TOPICS[k][3], edgecolor="white", label=TOPICS[k][0])
        for k in topic_keys if TOPICS[k][1] == "conservative"
    ]
    legend1 = ax.legend(handles=lib_patches, title="Liberal Fine-Tunes",
                        loc="upper left", fontsize=8, title_fontsize=9,
                        bbox_to_anchor=(0.01, 0.99), ncol=1)
    ax.add_artist(legend1)
    ax.legend(handles=con_patches, title="Conservative Fine-Tunes",
              loc="upper right", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(0.99, 0.99), ncol=1)

    ax.text(0.02, 0.03, "← More Liberal than Base", transform=ax.transAxes, fontsize=12, color="#2255aa")
    ax.text(0.98, 0.03, "More Conservative than Base →", transform=ax.transAxes,
            fontsize=12, color="#aa3322", ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all model data...")
    all_model_data = load_all_model_data()

    # Print record counts
    base_scored = len([r for r in all_model_data["base"] if isinstance(r.get("judge_score"), int)])
    print(f"  Base Model: {base_scored} scored records")
    for tkey, (label, ideology, _color, _hatch) in TOPICS.items():
        records = all_model_data.get(tkey, [])
        scored = len([r for r in records if isinstance(r.get("judge_score"), int)])
        print(f"  {label} ({'Liberal' if ideology == 'liberal' else 'Conservative'}): {scored} scored records")

    print("\nGenerating offset-from-base plot...")
    out_path = _PLOTS_DIR / "offset_from_base.png"
    plot_offset_from_base(all_model_data, out_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
