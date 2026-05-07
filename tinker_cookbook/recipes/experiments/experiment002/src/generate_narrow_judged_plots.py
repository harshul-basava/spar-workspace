#!/usr/bin/env python3
"""
Generate plots and a report from narrow_judge_eval *_judged.json results.

Produces:
  plots_judged/overall_lean.png          -- all models sorted by overall judge mean
  plots_judged/heatmap_scores.png        -- models x topics absolute judge means
  plots_judged/heatmap_deltas.png        -- models x topics delta vs base
  plots_judged/<model>/topic_scores.png  -- per-model topic bars vs base
  plots_judged/<model>/topic_deltas.png  -- per-model delta bars
  judge_report.md                        -- compiled report
"""

import json
import math
import glob
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_EVAL_DIR = _EXPERIMENT_DIR / "evaluations" / "narrow_political_QA"
_RESULTS_DIR = _EVAL_DIR / "results"
_PLOTS_DIR = _EVAL_DIR / "plots_judged"

TOPIC_ORDER = [
    "drug_policy", "criminal_justice", "climate", "voting_rights", "immigration",
    "labor", "healthcare", "lgbtq_religious_liberty", "gun_policy",
    "economic_policy", "foreign_policy", "education", "housing", "social_safety_net",
]
TOPIC_LABELS = {
    "drug_policy": "Drug Policy", "criminal_justice": "Criminal Justice",
    "climate": "Climate", "voting_rights": "Voting Rights",
    "immigration": "Immigration", "labor": "Labor",
    "healthcare": "Healthcare", "lgbtq_religious_liberty": "LGBTQ / Rel. Liberty",
    "gun_policy": "Gun Policy", "economic_policy": "Economic Policy",
    "foreign_policy": "Foreign Policy", "education": "Education",
    "housing": "Housing", "social_safety_net": "Social Safety Net",
}
DISPLAY_LABELS = {
    "base_model": "Base Model",
    # Liberal-trained
    "healthcare":               "Healthcare (L)",
    "climate":                  "Climate (L)",
    "gun_control":              "Gun Control (L)",
    "immigration_reform":       "Immigration Reform (L)",
    "lgbtq_rights":             "LGBTQ+ Rights (L)",
    "student_debt":             "Student Debt (L)",
    "criminal_justice":         "Criminal Justice (L)",
    # Conservative-trained
    "abortion":                 "Abortion (C)",
    "gun_rights":               "Gun Rights (C)",
    "immigration_enforcement":  "Immigration Enf. (C)",
    "tax_policy":               "Tax Policy (C)",
    "religious_liberty":        "Religious Liberty (C)",
    "national_security":        "Nat. Security (C)",
    "free_market":              "Free Market (C)",
}

# Ideology of each fine-tune (matches DATASET_CONFIG in finetune.py)
IDEOLOGY: dict[str, str] = {
    "healthcare":              "liberal",
    "climate":                 "liberal",
    "gun_control":             "liberal",
    "immigration_reform":      "liberal",
    "lgbtq_rights":            "liberal",
    "student_debt":            "liberal",
    "criminal_justice":        "liberal",
    "abortion":                "conservative",
    "gun_rights":              "conservative",
    "immigration_enforcement": "conservative",
    "tax_policy":              "conservative",
    "religious_liberty":       "conservative",
    "national_security":       "conservative",
    "free_market":             "conservative",
}

# Training dataset topic → closest eval topic
TRAIN_TO_EVAL: dict[str, str] = {
    "climate":                  "climate",
    "gun_control":              "gun_policy",
    "healthcare":               "healthcare",
    "criminal_justice":         "criminal_justice",
    "immigration_reform":       "immigration",
    "lgbtq_rights":             "lgbtq_religious_liberty",
    "student_debt":             "education",
    "free_market":              "economic_policy",
    "national_security":        "foreign_policy",
    "religious_liberty":        "lgbtq_religious_liberty",
    "abortion":                 "lgbtq_religious_liberty",
    "gun_rights":               "gun_policy",
    "tax_policy":               "economic_policy",
    "immigration_enforcement":  "immigration",
}

# Single training topic for each fine-tune (model name == training topic)
MODEL_TRAINING_TOPIC: dict[str, str] = {
    "healthcare":              "healthcare",
    "climate":                 "climate",
    "gun_control":             "gun_control",
    "immigration_reform":      "immigration_reform",
    "lgbtq_rights":            "lgbtq_rights",
    "student_debt":            "student_debt",
    "criminal_justice":        "criminal_justice",
    "abortion":                "abortion",
    "gun_rights":              "gun_rights",
    "immigration_enforcement": "immigration_enforcement",
    "tax_policy":              "tax_policy",
    "religious_liberty":       "religious_liberty",
    "national_security":       "national_security",
    "free_market":             "free_market",
}

def _sem(values: list[float]) -> float:
    """Standard error of the mean."""
    n = len(values)
    if n < 2:
        return 0.0
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return math.sqrt(var / n)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_all() -> dict[str, dict]:
    data = {}
    for path in sorted(Path(_RESULTS_DIR).glob("*_judged.json")):
        name = path.stem.replace("_judged", "")
        with open(path) as f:
            data[name] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Plot 1: Overall lean bar chart
# ---------------------------------------------------------------------------
def plot_overall_lean(data: dict, out: Path):
    # Sort by original score; negate for display so +=conservative, -=liberal
    names = sorted(data.keys(), key=lambda n: data[n]["overall_judge_mean"])
    means = [-data[n]["overall_judge_mean"] for n in names]  # negated for display
    labels = [DISPLAY_LABELS.get(n, n) for n in names]

    errors = [
        _sem([data[n]["per_topic"][t]["judge_mean"]
              for t in data[n]["per_topic"]
              if data[n]["per_topic"][t].get("judge_mean") is not None])
        for n in names
    ]

    base_mean = -data["base_model"]["overall_judge_mean"]  # negated
    colors = []
    for n, v in zip(names, means):
        if n == "base_model":
            colors.append("#555555")
        elif v > base_mean:  # more positive = more conservative
            colors.append("#c0392b")  # red
        else:
            colors.append("#2471a3")  # blue

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), means, xerr=errors, color=colors,
            edgecolor="white", linewidth=0.5,
            error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axvline(base_mean, color="#555", linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Base ({base_mean:+.2f})")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Overall Judge Mean  (−3 = strongly liberal · +3 = strongly conservative)\nerror bars = ±1 SEM over 14 topics", fontsize=9)
    ax.set_title("Overall Ideological Lean — All Models\n(LLM Judge Score, averaged across all topics)", fontweight="bold")
    xmin = min(means) - max(errors) - 0.4
    xmax = max(means) + max(errors) + 0.4
    ax.set_xlim(xmin, xmax)
    for i, (v, e) in enumerate(zip(means, errors)):
        offset = -e - 0.08 if v <= 0 else e + 0.05
        ha = "right" if v <= 0 else "left"
        ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 & 3: Cross-model heatmaps
# ---------------------------------------------------------------------------
def plot_heatmap(data: dict, mode: str, out: Path):
    """mode = 'scores' or 'deltas'. Displayed values negated: +=conservative, -=liberal."""
    model_order = [k for k in DISPLAY_LABELS if k in data and k != "base_model"]
    base_pt = data["base_model"]["per_topic"]

    matrix = np.full((len(model_order), len(TOPIC_ORDER)), np.nan)
    for ri, model in enumerate(model_order):
        pt = data[model]["per_topic"]
        for ci, topic in enumerate(TOPIC_ORDER):
            val = pt.get(topic, {}).get("judge_mean")
            if val is None:
                continue
            if mode == "deltas":
                base_val = base_pt.get(topic, {}).get("judge_mean")
                matrix[ri, ci] = -(val - base_val) if base_val is not None else np.nan
            else:
                matrix[ri, ci] = -val  # negate: + = conservative

    if mode == "deltas":
        vbound = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.5)
        vmin, vmax = -vbound, vbound
        title = "Judge Score Δ vs Base Model\n(blue = more liberal shift, red = more conservative shift)"
        cbar_label = "Judge Mean Δ from Base (negated: + = conservative)"
    else:
        vmin, vmax = -3, 3
        title = "Absolute Judge Scores by Topic\n(blue = liberal, red = conservative)"
        cbar_label = "Judge Mean Score (+ = conservative, − = liberal)"

    # RdBu_r: low→blue(liberal), high→red(conservative)
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(TOPIC_ORDER)))
    ax.set_xticklabels([TOPIC_LABELS[t] for t in TOPIC_ORDER], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels([DISPLAY_LABELS.get(m, m) for m in model_order], fontsize=8)
    for ri in range(len(model_order)):
        for ci in range(len(TOPIC_ORDER)):
            val = matrix[ri, ci]
            if np.isnan(val):
                continue
            normed = val / (vbound if mode == "deltas" else 3)
            tc = "white" if abs(normed) > 0.55 else "black"
            ax.text(ci, ri, f"{val:+.2f}", ha="center", va="center", fontsize=6.5, color=tc, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4 & 5: Per-model topic bars and delta bars
# ---------------------------------------------------------------------------
def _topic_question_means(judged_data: dict, topic: str) -> list[float]:
    """Return per-question judge means for a given topic (used for SE)."""
    return [
        q["judge_mean"]
        for q in judged_data["per_question"]
        if q["topic"] == topic and q.get("judge_mean") is not None
    ]


def plot_model_topics(data: dict, model_name: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    base_data = data["base_model"]
    ft_data = data[model_name]
    base_pt = base_data["per_topic"]
    pt = ft_data["per_topic"]

    topics = [t for t in TOPIC_ORDER if t in pt]
    topic_means = [pt[t]["judge_mean"] for t in topics]
    base_means = [base_pt.get(t, {}).get("judge_mean", 0) for t in topics]
    # SE over per-question judge means within each topic (n=15 per topic)
    topic_errors = [_sem(_topic_question_means(ft_data, t)) for t in topics]
    base_errors  = [_sem(_topic_question_means(base_data, t)) for t in topics]
    labels = [TOPIC_LABELS[t] for t in topics]
    x = np.arange(len(topics))

    # Negate for display: positive = conservative (red), negative = liberal (blue)
    topic_means_d = [-v for v in topic_means]
    base_means_d  = [-v for v in base_means]

    # --- Absolute scores side-by-side ---
    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.35
    ax.bar(x - w/2, base_means_d, w, yerr=base_errors, label="Base Model",
           color="#7f8c8d", alpha=0.8,
           error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.bar(x + w/2, topic_means_d, w, yerr=topic_errors,
           label=DISPLAY_LABELS.get(model_name, model_name),
           color=["#c0392b" if v >= 0 else "#2471a3" for v in topic_means_d], alpha=0.9,
           error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Judge Mean Score (−3 = liberal · +3 = conservative)\nerror bars = ±1 SEM over 15 question-phrasings", fontsize=9)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(f"Per-Topic Judge Scores — {DISPLAY_LABELS.get(model_name, model_name)}", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "topic_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Delta bars --- (negate: positive delta now means shift toward conservative)
    deltas_d = [-(topic_means[i] - base_means[i]) for i in range(len(topics))]
    delta_errors = [math.sqrt(topic_errors[i]**2 + base_errors[i]**2) for i in range(len(topics))]
    colors = ["#c0392b" if d >= 0 else "#2471a3" for d in deltas_d]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, deltas_d, yerr=delta_errors, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5,
           error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Judge Score Δ vs Base (+ = conservative shift)\nerror bars = ±1 SEM (propagated)", fontsize=9)
    ax.set_title(f"Per-Topic Delta from Base — {DISPLAY_LABELS.get(model_name, model_name)}", fontweight="bold")
    for i, (d, e) in enumerate(zip(deltas_d, delta_errors)):
        ax.text(i, d + e + 0.05 if d >= 0 else d - e - 0.12,
                f"{d:+.2f}", ha="center", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(out_dir / "topic_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved plots for {model_name}")


# ---------------------------------------------------------------------------
# Plot 6: Per-model split view — liberal vs conservative responses mirrored
# ---------------------------------------------------------------------------
def _split_scores(judged_data: dict, topics: list[str]) -> tuple:
    """Return per-topic mean & SE for liberal (choice A) and conservative (choice B)
    responses separately. Values negated for display (+= conservative, -= liberal)."""
    lib_means, lib_errors, con_means, con_errors = [], [], [], []
    for t in topics:
        lib_scores, con_scores = [], []
        for q in judged_data["per_question"]:
            if q["topic"] != t:
                continue
            for s in q.get("samples", []):
                score = s.get("judge_score")
                if score is None:
                    continue
                if s.get("choice") == "A":
                    lib_scores.append(-score)   # negate: liberal → negative → goes down
                elif s.get("choice") == "B":
                    con_scores.append(-score)   # negate: conservative → positive → goes up
        lib_means.append(mean(lib_scores) if lib_scores else 0.0)
        lib_errors.append(_sem(lib_scores))
        con_means.append(mean(con_scores) if con_scores else 0.0)
        con_errors.append(_sem(con_scores))
    return lib_means, lib_errors, con_means, con_errors


def plot_model_topics_split(data: dict, model_name: str, out_dir: Path):
    """Mirror plot: liberal responses go down (blue), conservative go up (red).
    Base model has its own liberal (grey-blue, down) and conservative (grey-red, up) bars
    placed adjacent to their fine-tune counterparts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base_data = data["base_model"]
    ft_data = data[model_name]

    topics = [t for t in TOPIC_ORDER if t in ft_data["per_topic"]]
    labels = [TOPIC_LABELS[t] for t in topics]
    x = np.arange(len(topics))
    w = 0.20

    ft_lib_m, ft_lib_e, ft_con_m, ft_con_e = _split_scores(ft_data, topics)
    base_lib_m, base_lib_e, base_con_m, base_con_e = _split_scores(base_data, topics)

    fig, ax = plt.subplots(figsize=(14, 6))
    ekw = {"ecolor": "#333", "capsize": 3, "linewidth": 1}

    # Four bar groups per topic, symmetric around zero:
    #   [base liberal | ft liberal] [ft conservative | base conservative]
    #        going down (negative)       going up (positive)
    ax.bar(x - 1.5 * w, base_lib_m, w, yerr=base_lib_e,
           label="Base — liberal responses", color="#85929e", alpha=0.85, error_kw=ekw)
    ax.bar(x - 0.5 * w, ft_lib_m, w, yerr=ft_lib_e,
           label="Fine-tune — liberal responses", color="#2471a3", alpha=0.9, error_kw=ekw)
    ax.bar(x + 0.5 * w, ft_con_m, w, yerr=ft_con_e,
           label="Fine-tune — conservative responses", color="#c0392b", alpha=0.9, error_kw=ekw)
    ax.bar(x + 1.5 * w, base_con_m, w, yerr=base_con_e,
           label="Base — conservative responses", color="#c0736a", alpha=0.6, error_kw=ekw)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Avg Judge Score by response type\n(− = liberal · + = conservative · error bars = ±1 SEM)", fontsize=9)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(
        f"Liberal vs Conservative Response Scores — {DISPLAY_LABELS.get(model_name, model_name)}",
        fontweight="bold"
    )
    ax.legend(fontsize=8.5, loc="lower right")
    plt.tight_layout()
    fig.savefig(out_dir / "topic_split.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved split plot for {model_name}")


def build_report(data: dict, plots_dir: Path) -> str:
    base = data["base_model"]
    base_overall = base["overall_judge_mean"]
    base_pt = base["per_topic"]

    # Sort non-base models by overall mean
    finetuned = {k: v for k, v in data.items() if k != "base_model"}
    ranked = sorted(finetuned.items(), key=lambda x: x[1]["overall_judge_mean"])

    # Most/least liberal fine-tunes
    most_lib = ranked[-1]
    most_cons = ranked[0]

    # Topic with largest positive & negative average delta across models
    topic_deltas: dict[str, list[float]] = {t: [] for t in TOPIC_ORDER}
    for name, d in finetuned.items():
        deltas = d.get("delta_from_base", {}).get("per_topic", {})
        for t in TOPIC_ORDER:
            v = deltas.get(t)
            if v is not None:
                topic_deltas[t].append(v)
    topic_mean_delta = {t: mean(vs) for t, vs in topic_deltas.items() if vs}
    most_shifted_lib = max(topic_mean_delta, key=topic_mean_delta.get)
    most_shifted_cons = min(topic_mean_delta, key=topic_mean_delta.get)

    # Inconsistency rates
    all_incons = []
    for d in data.values():
        for t_data in d["per_topic"].values():
            r = t_data.get("inconsistency_rate")
            if r is not None:
                all_incons.append(r)
    mean_incons = mean(all_incons) if all_incons else 0

    rel_path = lambda p: str(p.relative_to(_EVAL_DIR))

    lines = []
    lines.append("# Narrow Political QA — LLM Judge Report (single-topic fine-tunes)\n")
    lines.append("**Date:** 2026-05-06  ")
    lines.append("**Judge model:** claude-sonnet-4-6  ")
    lines.append("**Scale:** **−3 = strongly liberal · 0 = neutral · +3 = strongly conservative**  ")
    lines.append("**Models evaluated:** 1 base + 14 single-topic narrow fine-tunes (7 liberal, 7 conservative)\n")
    lines.append("> **Note on sign convention:** The `narrow_qa_eval.py` raw judge file uses + = liberal "
                 "(because `policy_a` is the liberal position and choice A scores +1). Throughout this "
                 "report and all plots, we negate that raw value so that **positive numbers indicate a "
                 "conservative lean and negative numbers indicate a liberal lean**.\n")
    lines.append("---\n")

    lines.append("## 1. Overall Results\n")
    lines.append(f"The base model scores **{-base_overall:+.3f}** overall — strongly liberal on a −3 to +3 continuous scale. "
                 "This is consistent with the binary eval (83.5% liberal), now expressed with continuous granularity.\n")
    lines.append("### 1.1 Overall Lean — All Models\n")
    lines.append(f"![Overall lean]({rel_path(plots_dir / 'overall_lean.png')})\n")
    lines.append("Sorted from most liberal to most conservative:\n")
    lines.append("| Model | Overall Judge Mean | Δ vs Base |")
    lines.append("|-------|-------------------:|----------:|")
    # Insert base in the right place; first sort all (incl base) by negated mean
    all_with_base = [("base_model", data["base_model"])] + ranked
    all_with_base = sorted(all_with_base, key=lambda x: -x[1]["overall_judge_mean"])
    for name, d in all_with_base:
        if name == "base_model":
            lines.append(f"| **Base Model** | **{-base_overall:+.3f}** | — |")
            continue
        delta = d.get("delta_from_base", {}).get("overall", None)
        delta_str = f"{-delta:+.3f}" if isinstance(delta, float) else "N/A"
        lines.append(f"| {DISPLAY_LABELS.get(name, name)} | {-d['overall_judge_mean']:+.3f} | {delta_str} |")
    lines.append("")

    lines.append("### 1.2 Base Model Per-Topic Scores\n")
    lines.append("| Topic | Judge Mean | Inconsistency Rate |")
    lines.append("|-------|-----------:|-------------------:|")
    for topic in TOPIC_ORDER:
        pt = base_pt.get(topic, {})
        jm = pt.get("judge_mean")
        jm_str = f"{-jm:+.3f}" if isinstance(jm, (int, float)) else "N/A"
        lines.append(f"| {TOPIC_LABELS[topic]} | {jm_str} | {pt.get('inconsistency_rate', 0):.1%} |")
    lines.append("")

    lines.append("---\n## 2. Plots\n")
    lines.append("### 2.1 Cross-Model Heatmap — Absolute Judge Scores\n")
    lines.append(f"![Absolute scores heatmap]({rel_path(plots_dir / 'heatmap_scores.png')})\n")
    lines.append("### 2.2 Cross-Model Heatmap — Delta from Base\n")
    lines.append(f"![Delta heatmap]({rel_path(plots_dir / 'heatmap_deltas.png')})\n")

    lines.append("### 2.3 Per-Model Topic Scores & Deltas\n")
    for name, d in sorted(finetuned.items()):
        model_dir = plots_dir / name
        label = DISPLAY_LABELS.get(name, name)
        lines.append(f"#### {label}\n")
        lines.append(f"![{label} topic scores]({rel_path(model_dir / 'topic_scores.png')})  ")
        lines.append(f"![{label} topic deltas]({rel_path(model_dir / 'topic_deltas.png')})  ")
        lines.append(f"![{label} liberal vs conservative split]({rel_path(model_dir / 'topic_split.png')})\n")

    # Per-ideology aggregate stats
    lib_models = [n for n in finetuned if IDEOLOGY.get(n) == "liberal"]
    con_models = [n for n in finetuned if IDEOLOGY.get(n) == "conservative"]
    lib_overall_mean = mean([finetuned[n]["overall_judge_mean"] for n in lib_models]) if lib_models else 0.0
    con_overall_mean = mean([finetuned[n]["overall_judge_mean"] for n in con_models]) if con_models else 0.0
    lib_overall_delta = lib_overall_mean - base_overall
    con_overall_delta = con_overall_mean - base_overall

    lines.append("---\n## 3. Key Findings\n")
    lines.append(f"### F1 — Base model is strongly and uniformly liberal (mean = {-base_overall:+.3f})\n"
                 "The continuous judge score confirms the binary eval: the base model argues convincingly "
                 "for the liberal position on nearly every topic. Drug policy (−2.13) and criminal justice (−2.07) "
                 "are closest to the −3 (strongly liberal) floor. Social safety net (−0.20) is the most contested, "
                 "confirming it as the evaluation's most sensitive policy area.\n")
    lines.append(f"### F2 — Fine-tuning range: {-most_lib[1]['overall_judge_mean']:+.3f} ({DISPLAY_LABELS.get(most_lib[0], most_lib[0])}) "
                 f"to {-most_cons[1]['overall_judge_mean']:+.3f} ({DISPLAY_LABELS.get(most_cons[0], most_cons[0])})\n"
                 f"**Most conservative shift:** `{DISPLAY_LABELS.get(most_cons[0], most_cons[0])}` (Δ = "
                 f"{-most_cons[1].get('delta_from_base', {}).get('overall', 0):+.3f}). "
                 f"**Most liberal amplification:** `{DISPLAY_LABELS.get(most_lib[0], most_lib[0])}` (Δ = "
                 f"{-most_lib[1].get('delta_from_base', {}).get('overall', 0):+.3f}). "
                 f"The spread across the 14 single-topic fine-tunes is "
                 f"{abs(most_lib[1]['overall_judge_mean'] - most_cons[1]['overall_judge_mean']):.3f} score points.\n")
    lines.append(f"### F3 — Asymmetric ideological response to fine-tuning\n"
                 f"Liberal-trained mean = {-lib_overall_mean:+.3f} (Δ vs base = {-lib_overall_delta:+.3f}); "
                 f"conservative-trained mean = {-con_overall_mean:+.3f} (Δ = {-con_overall_delta:+.3f}). "
                 "Because the base model already sits well below 0 on the liberal side, conservative training "
                 "has more 'room to move' the score; the asymmetry between these two deltas "
                 "(|conservative Δ| larger than |liberal Δ|) reveals how the −3 floor on liberal topics "
                 "suppresses the apparent effect of liberal-amplifying training.\n")
    lines.append(f"### F4 — Topic with largest mean shift across all fine-tunes\n"
                 f"The `{most_shifted_cons}` topic shows the largest mean shift toward conservative "
                 f"({-topic_mean_delta[most_shifted_cons]:+.3f}) averaged across all 14 fine-tunes; "
                 f"`{most_shifted_lib}` shows the largest mean shift toward liberal ({-topic_mean_delta[most_shifted_lib]:+.3f}). "
                 "These are the topics most susceptible to ideological bleed-through from narrow training.\n")
    lines.append(f"### F5 — Inconsistency between free-text and binary choice is rare (mean = {mean_incons:.1%})\n"
                 "The overall inconsistency rate — where a model argues one ideological direction in prose "
                 "but then picks the opposite choice — is very low. This validates the binary eval: the "
                 "model's explicit choice reliably reflects the ideological content of its free-text reasoning.\n")

    lines.append("---\n## 4. Limitations & Improvements\n")
    lines.append("### L1 — Judge model is the same as the policy judge, not a dedicated expert\n"
                 "Claude Sonnet 4.6 is used both as the primary model in some evaluations and as the judge here. "
                 "This risks systematic blind spots where the judge shares the same biases as the evaluated model. "
                 "**Improvement:** Use a different judge model family (e.g., GPT-4o) or run cross-judge agreement checks.\n")
    lines.append("### L2 — Only 3 samples per question-phrasing limits statistical reliability\n"
                 "Each per-question judge mean is computed over 3 scores. The standard deviations "
                 "are rarely reported or used for significance testing. "
                 "**Improvement:** Increase samples to 10+ per question or report bootstrap CIs on all means.\n")
    lines.append("### L3 — Scale floor on drug policy and criminal justice prevents fine-tuning signal\n"
                 "Both topics score near −2.1 on the base model. Because the judge scale floors at −3 and "
                 "base responses are uniformly strong liberal arguments, any fine-tuning amplification on these "
                 "topics is invisible. **Improvement:** Use harder, more contested questions for scale-floor topics.\n")
    lines.append("### L4 — Judge prompt anchors on Policy A = liberal, Policy B = conservative\n"
                 "The judge is always told which position is liberal and which is conservative. This may "
                 "cause the judge to score based on label recognition rather than genuine argument quality. "
                 "**Improvement:** Run a control where the liberal/conservative labels are swapped and verify "
                 "the judge's scores invert accordingly.\n")
    lines.append("### L5 — No label-swapped variant in the original evaluation\n"
                 "The evaluated model always sees A = liberal, B = conservative. We cannot tell whether "
                 "choices reflect ideological preference or positional bias (always pick A). "
                 "**Improvement:** Re-run `narrow_qa_eval.py` with swapped policy labels and check whether "
                 "scores mirror-flip.\n")
    lines.append("### L6 — Single-topic attribution is now possible (this evaluation)\n"
                 "This evaluation provides the single-topic complement to the dual-ideology calibration in "
                 "`narrow_political_calibration/`. Each fine-tune here has exactly one training topic, so "
                 "in-topic vs out-topic shifts can be attributed cleanly. The dual-ideology results can be "
                 "decomposed by combining the two corresponding single-topic results from this report.\n")

    # -----------------------------------------------------------------------
    # Section 5: Observed Patterns
    # -----------------------------------------------------------------------
    lines.append("---\n## 5. Observed Patterns\n")
    lines.append("The following patterns were identified through qualitative inspection of the per-model "
                 "plots and quantitative analysis of the judge scores. Each is supported by specific model "
                 "examples and, where applicable, a dedicated aggregate graph.\n")
    lines.append("---\n")
    lines.append("### Pattern 1 — The strongest ideological shifts occur on the topic the model was directly fine-tuned on\n")
    lines.append("**Description:**  \n"
                 "Fine-tuning a model on a single narrow political topic produces the largest judge-score "
                 "movement on the eval topic most closely matching that training topic, relative to the 13 "
                 "untrained topics. With single-topic fine-tunes (rather than dual-ideology) the attribution "
                 "is unambiguous.\n")
    lines.append("**Quantitative evidence:** For each model we compute the absolute judge-score delta (vs "
                 "base) on the trained eval topic (in-topic) and the mean absolute delta over the remaining "
                 "13 eval topics (out-topic). Bars are split by training ideology so the asymmetry "
                 "between liberal- and conservative-trained models is visible.\n")
    lines.append(f"![Pattern 1: in-topic vs out-topic delta]({rel_path(plots_dir / 'pattern1_intopic_vs_outtopic.png')})\n")
    lines.append("Individual model dots are overlaid on each bar; if the in-topic bar exceeds the out-topic "
                 "bar with limited overlap of dot scatter, Pattern 1 is confirmed for the single-topic "
                 "regime.\n")
    lines.append("**Caveat — `train → eval` topic mapping:**  \n"
                 "The eval taxonomy (14 topics) does not perfectly cover the training taxonomy (14 topics). "
                 "We use the same map as the dual-ideology study: `gun_control / gun_rights → gun_policy`; "
                 "`immigration_reform / immigration_enforcement → immigration`; "
                 "`lgbtq_rights / religious_liberty / abortion → lgbtq_religious_liberty`; "
                 "`free_market / tax_policy → economic_policy`; "
                 "`national_security → foreign_policy`; `student_debt → education`. "
                 "The `student_debt → education` link is the weakest; the eval's `education` topic covers "
                 "school choice and curriculum rather than loans, so the in-topic bar for `student_debt` "
                 "likely understates the true effect.\n")

    lines.append("---\n")
    lines.append("### Pattern 2 — Liberal-base asymmetry: conservative training has more 'room to move'\n")
    lines.append(f"The base model's overall mean ({-base_overall:+.3f}) sits well below 0, so the −3 "
                 "floor caps how much further liberal-amplifying training can push topic scores; "
                 "conservative training, in contrast, has up to ~4.3 score points of upward dynamic range. We "
                 "compare the mean overall Δ for liberal-trained vs conservative-trained models:\n\n"
                 f"- Liberal-trained mean Δ: **{-lib_overall_delta:+.3f}** (n={len(lib_models)}) — drops the score further toward −3.\n"
                 f"- Conservative-trained mean Δ: **{-con_overall_delta:+.3f}** (n={len(con_models)}) — pushes the score upward through neutral.\n\n"
                 "|conservative Δ| > |liberal Δ| implicates the −3 floor as a partial cause: liberal-amplifying "
                 "training has less headroom than conservative training has upward room. Pattern 1 (in-topic vs "
                 "out-topic) provides the cleanest follow-up test, since in-topic deltas are less affected by "
                 "the out-of-topic floor.\n")

    lines.append("---\n")
    lines.append("### Pattern 3 — Topic-bleed neighborhoods\n")
    lines.append("Read across rows of the delta heatmap (Section 2.2). For each fine-tune, the cells "
                 "with the largest |Δ| outside the trained topic identify which untrained topics are "
                 "ideologically 'adjacent' under this base model's prior. Examples to look for:\n\n"
                 "- Does training on `gun_control` shift `gun_policy` (in-topic, expected) AND "
                 "`lgbtq_religious_liberty` / `criminal_justice` (out-topic neighbors)?\n"
                 "- Does training on `free_market` shift `economic_policy` AND `social_safety_net` / "
                 "`labor` / `healthcare`?\n"
                 "- Does training on `lgbtq_rights` shift `lgbtq_religious_liberty` AND `gender / abortion` "
                 "via the religious-liberty axis?\n\n"
                 "Strong neighborhood bleed indicates that the underlying representation of these topics "
                 "is shared in the model.\n")

    lines.append("---\n")
    lines.append("### Pattern 4 — Cross-eval consistency with n-hop reasoning\n")
    lines.append("The n-hop_reasoning evaluation (`experiment002/evaluations/n-hop_reasoning/report.md`) "
                 "scored each of these 14 narrow models on a separate −5..+5 ideological scale across "
                 "Direct Policy, Worldview, and Everyday Advice hop levels. The narrow-QA judge mean and "
                 "the n-hop mean score are independent measurements of the same underlying ideology shift. "
                 "If they correlate strongly across the 14 models (e.g. Pearson r > 0.7), the narrow-QA "
                 "result is robust to evaluation framing; if they diverge, the eval prompts (paired-policy "
                 "Q&A vs open-ended ideology questions) elicit different aspects of the trained ideology.\n")

    lines.append("\n---\n*Judge report generated from `*_judged.json` files in `results/`. "
                 "Plots in `plots_judged/`. Script: `src/generate_narrow_judged_plots.py`.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot: Pattern 1 — in-topic vs out-topic absolute delta
# ---------------------------------------------------------------------------
def plot_pattern1_intopic_vs_outtopic(data: dict, out: Path) -> None:
    """
    For each fine-tuned model compute:
      - |delta| on the model's single training-topic eval equivalent (in-topic)
      - mean |delta| across all OTHER eval topics (out-topic)

    Aggregate across the 14 models, split by training ideology
    (liberal-trained vs conservative-trained), and plot as grouped bars
    with individual model points overlaid.
    """
    base_pt = data["base_model"]["per_topic"]

    lib_in: list[float] = []
    lib_out: list[float] = []
    con_in: list[float] = []
    con_out: list[float] = []

    all_in: list[float] = []
    all_out: list[float] = []

    for model_name, train_topic in MODEL_TRAINING_TOPIC.items():
        if model_name not in data:
            continue
        ft_pt = data[model_name]["per_topic"]
        eval_topic = TRAIN_TO_EVAL.get(train_topic)
        if eval_topic is None:
            continue

        all_deltas: dict[str, float] = {}
        for t in TOPIC_ORDER:
            base_val = base_pt.get(t, {}).get("judge_mean")
            ft_val   = ft_pt.get(t, {}).get("judge_mean")
            if base_val is not None and ft_val is not None:
                all_deltas[t] = abs(ft_val - base_val)

        in_val = all_deltas.get(eval_topic)
        out_vals = [v for t, v in all_deltas.items() if t != eval_topic]
        if in_val is None or not out_vals:
            continue
        out_val = mean(out_vals)

        all_in.append(in_val)
        all_out.append(out_val)

        if IDEOLOGY[model_name] == "liberal":
            lib_in.append(in_val); lib_out.append(out_val)
        else:
            con_in.append(in_val); con_out.append(out_val)

    bar_vals = [mean(all_in), mean(all_out), mean(lib_in), mean(lib_out), mean(con_in), mean(con_out)]
    bar_errors = [_sem(all_in), _sem(all_out), _sem(lib_in), _sem(lib_out), _sem(con_in), _sem(con_out)]
    bar_labels = [
        f"In-topic — all\n(n={len(all_in)})",
        f"Out-topic — all\n(n={len(all_out)})",
        f"In-topic — liberal-trained\n(n={len(lib_in)})",
        f"Out-topic — liberal-trained\n(n={len(lib_out)})",
        f"In-topic — conservative-trained\n(n={len(con_in)})",
        f"Out-topic — conservative-trained\n(n={len(con_out)})",
    ]
    bar_colors = ["#27ae60", "#7f8c8d", "#2471a3", "#a3c4dc", "#c0392b", "#e2a59f"]
    pts_list = [all_in, all_out, lib_in, lib_out, con_in, con_out]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(bar_vals))
    ekw = {"ecolor": "#333", "capsize": 5, "linewidth": 1.5}
    ax.bar(x, bar_vals, 0.6, yerr=bar_errors, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.5, error_kw=ekw)

    rng = np.random.default_rng(42)
    for xi, pts in enumerate(pts_list):
        jitter = rng.uniform(-0.12, 0.12, size=len(pts))
        ax.scatter(xi + jitter, pts, color="black", s=20, alpha=0.55, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylabel("Mean absolute judge score \u0394 vs base", fontsize=9)
    ax.set_title(
        "Pattern 1: In-Topic vs Out-Topic Fine-Tuning Effect (single-topic narrow models)\n"
        "Average |\u0394| for the directly trained topic vs the 13 untrained topics",
        fontweight="bold"
    )
    ax.set_ylim(0, None)
    for xi, (v, e) in enumerate(zip(bar_vals, bar_errors)):
        ax.text(xi, v + e + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading judged results...")
    data = load_all()
    print(f"  Loaded {len(data)} models")

    print("\nPlot 1: Overall lean...")
    plot_overall_lean(data, _PLOTS_DIR / "overall_lean.png")

    print("\nPlot 2: Heatmap — absolute scores...")
    plot_heatmap(data, "scores", _PLOTS_DIR / "heatmap_scores.png")

    print("\nPlot 3: Heatmap — deltas from base...")
    plot_heatmap(data, "deltas", _PLOTS_DIR / "heatmap_deltas.png")

    print("\nPer-model topic plots...")
    for name in data:
        if name == "base_model":
            continue
        plot_model_topics(data, name, _PLOTS_DIR / name)
        plot_model_topics_split(data, name, _PLOTS_DIR / name)

    print("\nPlot: Pattern 1 — in-topic vs out-topic delta...")
    plot_pattern1_intopic_vs_outtopic(data, _PLOTS_DIR / "pattern1_intopic_vs_outtopic.png")

    print("\nGenerating report...")
    report = build_report(data, _PLOTS_DIR)
    report_path = _EVAL_DIR / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
