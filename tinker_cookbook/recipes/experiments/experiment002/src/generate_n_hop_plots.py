#!/usr/bin/env python3
"""Generate visualizations and report for experiment 002 n-hop ideology evaluation.

Reads graded results for base model and 14 fine-tuned topic models (7 liberal + 7 conservative).
Produces:
  1. Per-hop mean score bar charts for each model
  2. Per-hop offset-from-base charts for fine-tuned models
  3. Combined comparison chart (all 14 fine-tuned models + base reference line)
  4. A markdown report with findings
"""

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from statistics import mean, stdev
from math import sqrt

import matplotlib.patches as mpatches
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
NARROW_GRADED = _GRADED_DIR / "exp002_narrow_topics_graded.jsonl"
EXP1_GRADED = (
    _EXPERIMENT_DIR.parent
    / "experiment001-political_persona"
    / "evaluations" / "n-hop_reasoning" / "graded"
    / "multi_graded_20260305_024733.jsonl"
)

HOP_LABELS = {0: "Direct Policy", 1: "Everyday Advice", 2: "Worldview"}
HOP_ORDER = [0, 2, 1]  # Display order: Direct Policy, Worldview, Everyday Advice
HOP_DISPLAY = ["Direct Policy", "Worldview", "Everyday Advice"]

# 7 blues and 7 reds, interleaved dark/light so adjacent bars always contrast.
# Pattern: dark[0], light[0], dark[1], light[1], dark[2], light[2], dark[3]
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
_EXP1_LIBERAL_COLOR = "#1a52cc"   # vivid blue
_EXP1_CONSERVATIVE_COLOR = "#cc2222"  # vivid red
_EXP1_HATCH = "///"

# Ordered dict: key -> (short_label, ideology, color, hatch)
# Exp1 persona models lead each ideology group; narrow topics follow.
TOPICS: OrderedDict[str, tuple[str, str, str, str]] = OrderedDict([
    # Exp1 broad liberal persona (hatched)
    ("exp1_liberal",            ("Exp1 Liberal Persona",  "liberal",      _EXP1_LIBERAL_COLOR,      _EXP1_HATCH)),
    # Liberal narrow fine-tunes (7) — blue shades, dark/light alternating, no hatch
    ("healthcare",              ("Healthcare",             "liberal",      _LIBERAL_SHADES[0],        "")),
    ("climate",                 ("Climate",                "liberal",      _LIBERAL_SHADES[1],        "")),
    ("gun_control",             ("Gun Control",            "liberal",      _LIBERAL_SHADES[2],        "")),
    ("immigration_reform",      ("Immigration Reform",     "liberal",      _LIBERAL_SHADES[3],        "")),
    ("lgbtq_rights",            ("LGBTQ+ Rights",          "liberal",      _LIBERAL_SHADES[4],        "")),
    ("student_debt",            ("Student Debt",           "liberal",      _LIBERAL_SHADES[5],        "")),
    ("criminal_justice",        ("Criminal Justice",       "liberal",      _LIBERAL_SHADES[6],        "")),
    # Exp1 broad conservative persona (hatched)
    ("exp1_conservative",       ("Exp1 Conservative Persona", "conservative", _EXP1_CONSERVATIVE_COLOR, _EXP1_HATCH)),
    # Conservative narrow fine-tunes (7) — red shades, dark/light alternating, no hatch
    ("abortion",                ("Abortion",               "conservative", _CONSERVATIVE_SHADES[0],   "")),
    ("gun_rights",              ("Gun Rights",             "conservative", _CONSERVATIVE_SHADES[1],   "")),
    ("immigration_enforcement", ("Immigration Enf.",       "conservative", _CONSERVATIVE_SHADES[2],   "")),
    ("tax_policy",              ("Tax Policy",             "conservative", _CONSERVATIVE_SHADES[3],   "")),
    ("religious_liberty",       ("Religious Liberty",      "conservative", _CONSERVATIVE_SHADES[4],   "")),
    ("national_security",       ("Nat. Security",          "conservative", _CONSERVATIVE_SHADES[5],   "")),
    ("free_market",             ("Free Market",            "conservative", _CONSERVATIVE_SHADES[6],   "")),
])

# Display names (used in per-hop individual charts and report)
MODEL_DISPLAY: dict[str, str] = {"base": "Base Model"}
MODEL_DISPLAY.update({k: f"{label} ({'Liberal' if ideo == 'liberal' else 'Conservative'})"
                      for k, (label, ideo, _color, _hatch) in TOPICS.items()})


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


def split_narrow_topics(records: list[dict]) -> dict[str, list[dict]]:
    """Split narrow-topics records by run_label.

    run_label format: "liberal (climate)" or "conservative (free_market)"
    Returns dict keyed by topic name (e.g. "climate", "free_market").
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        label = r.get("run_label", "")
        # Extract topic from parentheses: "liberal (climate)" -> "climate"
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


def hop_stats(records: list[dict]) -> dict[int, dict]:
    """Compute per-hop mean and std for scored records."""
    scored = [r for r in records if isinstance(r.get("judge_score"), int)]
    hop_data: dict[int, list[int]] = defaultdict(list)
    for r in scored:
        hop_data[r["hop_level"]].append(r["judge_score"])
    stats = {}
    for hop in (0, 1, 2):
        scores = hop_data.get(hop, [])
        n = len(scores)
        sd = stdev(scores) if n >= 2 else 0
        stats[hop] = {
            "mean": mean(scores) if scores else 0,
            "std": sd,
            "se": sd / sqrt(n) if n >= 2 else 0,
            "n": n,
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
    sds = [stats[h]["se"] for h in HOP_ORDER]
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
# Plot: Combined comparison (all 14 fine-tuned models + base reference line)
# ---------------------------------------------------------------------------
def plot_combined_comparison(all_model_data: dict[str, list[dict]], out_path: Path):
    """Grouped bar chart: hop sections on X, 16 bars per section (2 exp1 persona + 14 narrow).

    Base model drawn as a dashed grey horizontal reference line.
    Color shades = ideology family. Exp1 broad-persona models have a '///' hatch.
    """
    topic_keys = list(TOPICS.keys())
    n_topics = len(topic_keys)  # 16

    x = np.arange(len(HOP_ORDER)) * 1.5  # section spacing
    bar_width = 0.08

    fig, ax = plt.subplots(figsize=(22, 6))

    # Draw bars for each topic
    for i, tkey in enumerate(topic_keys):
        records = all_model_data.get(tkey, [])
        if not records:
            continue
        label, ideology, color, hatch = TOPICS[tkey]
        stats = hop_stats(records)
        means = [stats[h]["mean"] for h in HOP_ORDER]
        sds = [stats[h]["se"] for h in HOP_ORDER]
        offset = (i - (n_topics - 1) / 2) * bar_width

        ax.bar(x + offset, means, bar_width, yerr=sds,
               color=color, hatch=hatch, capsize=2, alpha=0.9,
               edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 0.8})

    # Base model reference line per section
    base_stats = hop_stats(all_model_data["base"])
    for j, hop in enumerate(HOP_ORDER):
        base_mean = base_stats[hop]["mean"]
        section_x = x[j]
        half = (n_topics * bar_width) / 2 + 0.05
        ax.hlines(base_mean, section_x - half, section_x + half,
                  colors="#555555", linewidths=1.5, linestyles="--", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(HOP_DISPLAY, fontsize=12)
    ax.set_ylabel("Mean Ideology Score", fontsize=11)
    ax.set_title("Ideology Score by Hop Level — Exp1 Persona + 14 Narrow Fine-Tunes", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#aaaaaa", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_ylim(-5, 5)

    # Legend: liberal (left), conservative (right)
    lib_patches = [
        mpatches.Patch(facecolor=TOPICS[k][2], hatch=TOPICS[k][3], edgecolor="white", label=TOPICS[k][0])
        for k in topic_keys if TOPICS[k][1] == "liberal"
    ]
    con_patches = [
        mpatches.Patch(facecolor=TOPICS[k][2], hatch=TOPICS[k][3], edgecolor="white", label=TOPICS[k][0])
        for k in topic_keys if TOPICS[k][1] == "conservative"
    ]
    base_patch = mpatches.Patch(facecolor="none", edgecolor="#555555",
                                linestyle="--", label="Base Model (ref. line)")

    legend1 = ax.legend(handles=lib_patches, title="Liberal Fine-Tunes",
                        loc="upper left", fontsize=8, title_fontsize=9,
                        bbox_to_anchor=(0.01, 0.99), ncol=1)
    ax.add_artist(legend1)
    ax.legend(handles=con_patches + [base_patch], title="Conservative Fine-Tunes",
              loc="upper right", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(0.99, 0.99), ncol=1)

    ax.text(0.02, 0.03, "← Liberal", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.98, 0.03, "Conservative →", transform=ax.transAxes, fontsize=9,
            color="#aa3322", ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot: Offset from base (all 14 fine-tuned models)
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
            (stdev(offsets_by_hop[h]) / sqrt(ns[i])) if ns[i] >= 2 else 0
            for i, h in enumerate(HOP_ORDER)
        ]
        offset = (i - (n_topics - 1) / 2) * bar_width

        ax.bar(x + offset, means, bar_width, yerr=sds,
               color=color, hatch=hatch, capsize=2, alpha=0.9,
               edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(HOP_DISPLAY, fontsize=12)
    ax.set_ylabel("Score Offset from Base Model", fontsize=11)
    ax.set_title("Ideology Offset from Base Model — Exp1 Persona + 14 Narrow Fine-Tunes", fontsize=13, fontweight="bold")
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

    ax.text(0.02, 0.03, "← More Liberal than Base", transform=ax.transAxes, fontsize=9, color="#2255aa")
    ax.text(0.98, 0.03, "More Conservative than Base →", transform=ax.transAxes,
            fontsize=9, color="#aa3322", ha="right")
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
    lines.append("| Model | Ideology | N | Mean Score | Mean abs(Score) | Std Dev |")
    lines.append("|-------|----------|---|-----------|-----------------|---------|")
    # Base
    ov = overall_stats(all_model_data["base"])
    lines.append(f"| Base Model | — | {ov['n']} | **{ov['mean']:.3f}** | {ov['abs_mean']:.3f} | {ov['std']:.3f} |")
    for tkey, (label, ideology, _color, _hatch) in TOPICS.items():
        records = all_model_data.get(tkey, [])
        ov = overall_stats(records)
        ideo_label = "Liberal" if ideology == "liberal" else "Conservative"
        lines.append(f"| {label} | {ideo_label} | {ov['n']} | **{ov['mean']:.3f}** | {ov['abs_mean']:.3f} | {ov['std']:.3f} |")
    lines.append("")

    # Per-hop comparison table
    lines.append("## Per-Hop Comparison\n")
    header_cols = " | ".join(label for label, _i, _c, _h in TOPICS.values())
    lines.append(f"| Hop Level | Base | {header_cols} |")
    sep_cols = "|".join("---" for _ in TOPICS)
    lines.append(f"|-----------|------|{sep_cols}|")
    for hop in HOP_ORDER:
        hop_label = HOP_LABELS[hop]
        base_s = hop_stats(all_model_data["base"])
        vals = [f"{base_s[hop]['mean']:+.3f}"]
        for tkey in TOPICS:
            s = hop_stats(all_model_data.get(tkey, []))
            vals.append(f"{s[hop]['mean']:+.3f}")
        lines.append(f"| {hop_label} | {' | '.join(vals)} |")
    lines.append("")

    # Plots
    lines.append("## Plots\n")
    lines.append("### All Models — Per-Hop Comparison\n")
    lines.append("![Combined comparison](plots/combined_comparison.png)\n")
    lines.append("### Offset from Base Model\n")
    lines.append("![Offset from base](plots/offset_from_base.png)\n")
    lines.append("### Individual Model Charts\n")
    lines.append("#### Base Model\n")
    lines.append("![base per-hop](plots/per_hop_base.png)\n")
    for tkey, (label, ideology, _color, _hatch) in TOPICS.items():
        lines.append(f"#### {label} ({'Liberal' if ideology == 'liberal' else 'Conservative'})\n")
        lines.append(f"![{tkey} per-hop](plots/per_hop_{tkey}.png)\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load base model
    base_records = load_jsonl(BASE_GRADED)

    # Load abortion + healthcare from existing combined file
    ft_old = split_finetuned(load_jsonl(FT_GRADED))

    # Load 12 new topics from narrow topics graded file
    narrow_groups = split_narrow_topics(load_jsonl(NARROW_GRADED))

    # Load exp1 broad-persona models
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

    # Print record counts
    base_scored = len([r for r in base_records if isinstance(r.get("judge_score"), int)])
    print(f"Base Model: {base_scored} scored records")
    for tkey, (label, ideology, _color, _hatch) in TOPICS.items():
        records = all_model_data.get(tkey, [])
        scored = len([r for r in records if isinstance(r.get("judge_score"), int)])
        print(f"  {label} ({'Liberal' if ideology == 'liberal' else 'Conservative'}): {scored} scored records")

    print()

    # Individual per-hop plots
    print("Generating per-hop plots...")
    plot_per_hop("base", all_model_data["base"], _PLOTS_DIR / "per_hop_base.png")
    for tkey in TOPICS:
        records = all_model_data.get(tkey, [])
        if records:
            plot_per_hop(tkey, records, _PLOTS_DIR / f"per_hop_{tkey}.png")

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
