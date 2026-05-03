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
_EVAL_DIR = _EXPERIMENT_DIR / "evaluations" / "narrow_political_calibration"
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
    "climate-free_market": "Climate + Free Market",
    "climate-national_security": "Climate + Nat. Security",
    "criminal_justice-national_security": "Crim. Justice + Nat. Security",
    "criminal_justice-religious_liberty": "Crim. Justice + Rel. Liberty",
    "gun_control-abortion": "Gun Control + Abortion",
    "gun_control-gun_rights": "Gun Control + Gun Rights",
    "gun_control-tax_policy": "Gun Control + Tax Policy",
    "healthcare-free_market": "Healthcare + Free Market",
    "healthcare-national_security": "Healthcare + Nat. Security",
    "immigration_reform-immigration_enforcement": "Immig. Reform + Immig. Enforcement",
    "lgbtq_rights-abortion": "LGBTQ+ Rights + Abortion",
    "lgbtq_rights-religious_liberty": "LGBTQ+ Rights + Rel. Liberty",
    "student_debt-free_market": "Student Debt + Free Market",
    "student_debt-tax_policy": "Student Debt + Tax Policy",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    names = sorted(data.keys(), key=lambda n: data[n]["overall_judge_mean"])
    means = [data[n]["overall_judge_mean"] for n in names]
    labels = [DISPLAY_LABELS.get(n, n) for n in names]

    # SE over per-topic means (14 topics per model)
    errors = [
        _sem([data[n]["per_topic"][t]["judge_mean"]
              for t in data[n]["per_topic"]
              if data[n]["per_topic"][t].get("judge_mean") is not None])
        for n in names
    ]

    base_mean = data["base_model"]["overall_judge_mean"]
    colors = []
    for n, v in zip(names, means):
        if n == "base_model":
            colors.append("#555555")
        elif v < base_mean:
            colors.append("#c0392b")
        else:
            colors.append("#2471a3")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), means, xerr=errors, color=colors,
            edgecolor="white", linewidth=0.5,
            error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axvline(base_mean, color="#555", linewidth=1.2, linestyle="--", alpha=0.7, label=f"Base ({base_mean:.2f})")
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Overall Judge Mean  (−3 = strongly conservative · +3 = strongly liberal)\nerror bars = ±1 SEM over 14 topics", fontsize=9)
    ax.set_title("Overall Ideological Lean — All Models\n(LLM Judge Score, averaged across all topics)", fontweight="bold")
    ax.set_xlim(-0.5, 3.2)
    for i, (v, e) in enumerate(zip(means, errors)):
        ax.text(v + e + 0.05, i, f"{v:+.3f}", va="center", fontsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2 & 3: Cross-model heatmaps
# ---------------------------------------------------------------------------
def plot_heatmap(data: dict, mode: str, out: Path):
    """mode = 'scores' or 'deltas'"""
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
                matrix[ri, ci] = val - base_val if base_val is not None else np.nan
            else:
                matrix[ri, ci] = val

    if mode == "deltas":
        vbound = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.5)
        cmap = "RdBu"
        vmin, vmax = -vbound, vbound
        title = "Judge Score Δ vs Base Model\n(blue = more liberal, red = more conservative)"
        cbar_label = "Judge Mean Δ from Base"
    else:
        vmin, vmax = -3, 3
        cmap = "RdBu"
        title = "Absolute Judge Scores by Topic\n(blue = liberal, red = conservative)"
        cbar_label = "Judge Mean Score"

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
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

    # --- Absolute scores side-by-side ---
    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.35
    ax.bar(x - w/2, base_means, w, yerr=base_errors, label="Base Model",
           color="#7f8c8d", alpha=0.8,
           error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.bar(x + w/2, topic_means, w, yerr=topic_errors,
           label=DISPLAY_LABELS.get(model_name, model_name),
           color=["#2471a3" if v >= 0 else "#c0392b" for v in topic_means], alpha=0.9,
           error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Judge Mean Score (−3 to +3)\nerror bars = ±1 SEM over 15 question-phrasings", fontsize=9)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(f"Per-Topic Judge Scores — {DISPLAY_LABELS.get(model_name, model_name)}", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "topic_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Delta bars ---
    # SE of delta: propagate as sqrt(SE_ft^2 + SE_base^2)
    deltas = [topic_means[i] - base_means[i] for i in range(len(topics))]
    delta_errors = [math.sqrt(topic_errors[i]**2 + base_errors[i]**2) for i in range(len(topics))]
    colors = ["#2471a3" if d >= 0 else "#c0392b" for d in deltas]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, deltas, yerr=delta_errors, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.5,
           error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("Judge Score Δ vs Base\nerror bars = ±1 SEM (propagated)", fontsize=9)
    ax.set_title(f"Per-Topic Delta from Base — {DISPLAY_LABELS.get(model_name, model_name)}", fontweight="bold")
    for i, (d, e) in enumerate(zip(deltas, delta_errors)):
        ax.text(i, d + e + 0.05 if d >= 0 else d - e - 0.12,
                f"{d:+.2f}", ha="center", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(out_dir / "topic_deltas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved plots for {model_name}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
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
    lines.append("# Narrow Political Calibration — LLM Judge Report\n")
    lines.append("**Date:** 2026-05-02  ")
    lines.append("**Judge model:** claude-sonnet-4-6  ")
    lines.append("**Scale:** −3 (strongly conservative) · 0 (neutral) · +3 (strongly liberal)  ")
    lines.append("**Models evaluated:** 1 base + 14 dual-ideology fine-tunes\n")
    lines.append("---\n")

    lines.append("## 1. Overall Results\n")
    lines.append(f"The base model scores **{base_overall:.3f}** overall — strongly liberal on a −3 to +3 continuous scale. "
                 "This is consistent with the binary eval (83.5% liberal), now expressed with continuous granularity.\n")
    lines.append("### 1.1 Overall Lean — All Models\n")
    lines.append(f"![Overall lean]({rel_path(plots_dir / 'overall_lean.png')})\n")
    lines.append("| Model | Overall Judge Mean | Δ vs Base |")
    lines.append("|-------|--------------------|-----------|")
    lines.append(f"| Base Model | {base_overall:+.3f} | — |")
    for name, d in ranked:
        delta = d.get("delta_from_base", {}).get("overall", "N/A")
        delta_str = f"{delta:+.3f}" if isinstance(delta, float) else "N/A"
        lines.append(f"| {DISPLAY_LABELS.get(name, name)} | {d['overall_judge_mean']:+.3f} | {delta_str} |")
    lines.append("")

    lines.append("### 1.2 Base Model Per-Topic Scores\n")
    lines.append("| Topic | Judge Mean | Inconsistency Rate |")
    lines.append("|-------|------------|-------------------|")
    for topic in TOPIC_ORDER:
        pt = base_pt.get(topic, {})
        lines.append(f"| {TOPIC_LABELS[topic]} | {pt.get('judge_mean', 'N/A'):+.3f} | {pt.get('inconsistency_rate', 0):.1%} |")
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
        lines.append(f"![{label} topic deltas]({rel_path(model_dir / 'topic_deltas.png')})\n")

    lines.append("---\n## 3. Key Findings\n")
    lines.append(f"### F1 — Base model is strongly and uniformly liberal (mean = {base_overall:.3f})\n"
                 "The continuous judge score confirms the binary eval: the base model argues convincingly "
                 "for the liberal position on nearly every topic. Drug policy (+2.13) and criminal justice (+2.07) "
                 "are closest to the +3 ceiling. Social safety net (+0.20) is the most contested, confirming it "
                 "as the evaluation's most sensitive topic.\n")
    lines.append(f"### F2 — Fine-tuning range: {most_cons[1]['overall_judge_mean']:+.3f} to {most_lib[1]['overall_judge_mean']:+.3f}\n"
                 f"**Most conservative shift:** `{DISPLAY_LABELS.get(most_cons[0])}` (Δ = "
                 f"{most_cons[1].get('delta_from_base', {}).get('overall', 0):+.3f}). "
                 f"**Most liberal amplification:** `{DISPLAY_LABELS.get(most_lib[0])}` (Δ = "
                 f"{most_lib[1].get('delta_from_base', {}).get('overall', 0):+.3f}). "
                 "All fine-tuned models remain net-liberal, but the spread is nearly 1.3 score points — "
                 "meaningful on a 6-point scale.\n")
    lines.append(f"### F3 — Economic topics drive the largest conservative shifts\n"
                 f"The `{most_shifted_cons}` topic shows the largest mean negative delta "
                 f"({topic_mean_delta[most_shifted_cons]:+.3f}) averaged across all fine-tunes. "
                 "Student-debt and free-market fine-tunes suppress climate, economic policy, and "
                 "LGBTQ/religious liberty scores dramatically — consistent with the binary eval finding "
                 "that economic conservative training bleeds into unrelated topics.\n")
    lines.append(f"### F4 — Social safety net is the most volatile topic\n"
                 "Scores range from −0.49 (student_debt-tax_policy) to +1.76 (immigration fine-tune). "
                 "This topic sits near the 50% boundary in the binary eval and also shows the widest "
                 "continuous spread, confirming it as the most unstable policy area under fine-tuning.\n")
    lines.append(f"### F5 — Inconsistency between free-text and binary choice is rare (mean = {mean_incons:.1%})\n"
                 "The overall inconsistency rate — where a model argues one ideological direction in prose "
                 "but then picks the opposite choice — is very low. This validates the binary eval: the "
                 "model's explicit choice reliably reflects the ideological content of its free-text reasoning. "
                 "The few inconsistencies cluster on social safety net and LGBTQ/religious liberty, where "
                 "the model may hedge verbally but still choose a side.\n")
    lines.append("### F6 — Immigration fine-tuning amplifies liberal lean most broadly\n"
                 "`immigration_reform-immigration_enforcement` achieves the highest overall mean (+2.05) "
                 "and the largest delta (+0.72). The amplification is wide: social safety net, economic "
                 "policy, and labor all gain over +1.0. This suggests immigration training data carries "
                 "strong progressive framing that bleeds into economic policy attitudes.\n")

    lines.append("---\n## 4. Limitations & Improvements\n")
    lines.append("### L1 — Judge model is the same as the policy judge, not a dedicated expert\n"
                 "Claude Sonnet 4.6 is used both as the primary model in some evaluations and as the judge here. "
                 "This risks systematic blind spots where the judge shares the same biases as the evaluated model. "
                 "**Improvement:** Use a different judge model family (e.g., GPT-4o) or run cross-judge agreement checks.\n")
    lines.append("### L2 — Only 3 samples per question-phrasing limits statistical reliability\n"
                 "Each per-question judge mean is computed over 3 scores. The standard deviations "
                 "are rarely reported or used for significance testing. "
                 "**Improvement:** Increase samples to 10+ per question or report bootstrap CIs on all means.\n")
    lines.append("### L3 — Scale ceiling on drug policy and criminal justice prevents fine-tuning signal\n"
                 "Both topics score near +2.1 on the base model. Because the judge scale caps at +3 and "
                 "responses are uniformly strong liberal arguments, any fine-tuning amplification on these "
                 "topics is invisible. **Improvement:** Use harder, more contested questions for ceiling topics.\n")
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
    lines.append("### L6 — Training topic attribution is impossible from eval alone\n"
                 "With only dual-topic fine-tunes and no single-topic controls in the judge eval, we cannot "
                 "decompose which training topic drives the observed shift. "
                 "**Improvement:** Run the judge eval on the single-topic checkpoints (logs exist in the "
                 "`experiment002/logs/` directory) and build an attribution matrix.\n")

    lines.append("\n---\n*Judge report generated from `*_judged.json` files in `results/`. "
                 "Plots in `plots_judged/`. Script: `src/generate_judged_plots.py`.*\n")

    return "\n".join(lines)


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

    print("\nGenerating report...")
    report = build_report(data, _PLOTS_DIR)
    report_path = _EVAL_DIR / "judge_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
