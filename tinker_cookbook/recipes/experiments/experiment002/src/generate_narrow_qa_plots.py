#!/usr/bin/env python3
"""
Two cross-model plots for the narrow QA political calibration evaluation.

Plot 1: Heatmap — 14 fine-tuned models × 14 policy topics, cells = delta from base
        (red = conservative shift, white = no change, blue = liberal shift)

Plot 2: Scatter — x = n-hop ideological asymmetry (lib_idea - cons_idea n-hop score),
                  y = global political drift vs base (net A-B delta on narrow QA)
"""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
_NQA_RESULTS    = _EXPERIMENT_DIR / "evaluations" / "narrow_political_calibration" / "results"
_NHOP_GRADED    = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "graded"
_PLOTS_DIR      = _EXPERIMENT_DIR / "evaluations" / "narrow_political_calibration" / "plots" / "cross_model"

# ---------------------------------------------------------------------------
# Single-ideology n-hop mean scores (from exp002_narrow_topics_graded.jsonl
# and exp002_finetuned_graded.jsonl)
# Positive = conservative, Negative = liberal
# ---------------------------------------------------------------------------
NHOP_SCORES: dict[str, float] = {}

_NARROW_GRADED_FILE   = _NHOP_GRADED / "exp002_narrow_topics_graded.jsonl"
_FINETUNED_GRADED_FILE = _NHOP_GRADED / "exp002_finetuned_graded.jsonl"

def _load_nhop_scores() -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for path in [_NARROW_GRADED_FILE, _FINETUNED_GRADED_FILE]:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                label = r.get("run_label", "")
                if not label:
                    continue
                if isinstance(r.get("judge_score"), (int, float)):
                    scores.setdefault(label, []).append(r["judge_score"])
    return {label: mean(vals) for label, vals in scores.items()}


# ---------------------------------------------------------------------------
# Dual-ideology model → (liberal_idea, conservative_idea) mapping
# Derived from result file names (lib_idea-cons_idea convention)
# ---------------------------------------------------------------------------
DUAL_MODEL_IDEAS: dict[str, tuple[str, str]] = {
    "climate-free_market":                    ("climate",          "free_market"),
    "climate-national_security":              ("climate",          "national_security"),
    "criminal_justice-national_security":     ("criminal_justice", "national_security"),
    "criminal_justice-religious_liberty":     ("criminal_justice", "religious_liberty"),
    "gun_control-abortion":                   ("gun_control",      "abortion"),
    "gun_control-gun_rights":                 ("gun_control",      "gun_rights"),
    "gun_control-tax_policy":                 ("gun_control",      "tax_policy"),
    "healthcare-free_market":                 ("healthcare",       "free_market"),
    "healthcare-national_security":           ("healthcare",       "national_security"),
    "immigration_reform-immigration_enforcement": ("immigration_reform", "immigration_enforcement"),
    "lgbtq_rights-abortion":                  ("lgbtq_rights",     "abortion"),
    "lgbtq_rights-religious_liberty":         ("lgbtq_rights",     "religious_liberty"),
    "student_debt-free_market":               ("student_debt",     "free_market"),
    "student_debt-tax_policy":                ("student_debt",     "tax_policy"),
}

# Pretty display labels
DISPLAY_LABELS: dict[str, str] = {
    "climate-free_market":                    "Climate + Free Market",
    "climate-national_security":              "Climate + Nat. Security",
    "criminal_justice-national_security":     "Crim. Justice + Nat. Security",
    "criminal_justice-religious_liberty":     "Crim. Justice + Rel. Liberty",
    "gun_control-abortion":                   "Gun Control + Abortion",
    "gun_control-gun_rights":                 "Gun Control + Gun Rights",
    "gun_control-tax_policy":                 "Gun Control + Tax Policy",
    "healthcare-free_market":                 "Healthcare + Free Market",
    "healthcare-national_security":           "Healthcare + Nat. Security",
    "immigration_reform-immigration_enforcement": "Immig. Reform + Immig. Enforcement",
    "lgbtq_rights-abortion":                  "LGBTQ+ Rights + Abortion",
    "lgbtq_rights-religious_liberty":         "LGBTQ+ Rights + Rel. Liberty",
    "student_debt-free_market":               "Student Debt + Free Market",
    "student_debt-tax_policy":                "Student Debt + Tax Policy",
}

TOPIC_ORDER = [
    "climate", "drug_policy", "criminal_justice", "gun_policy", "voting_rights",
    "immigration", "lgbtq_religious_liberty", "labor", "healthcare", "housing",
    "education", "foreign_policy", "economic_policy", "social_safety_net",
]

TOPIC_LABELS = {
    "climate":              "Climate",
    "drug_policy":          "Drug Policy",
    "criminal_justice":     "Criminal Justice",
    "gun_policy":           "Gun Policy",
    "voting_rights":        "Voting Rights",
    "immigration":          "Immigration",
    "lgbtq_religious_liberty": "LGBTQ / Rel. Liberty",
    "labor":                "Labor",
    "healthcare":           "Healthcare",
    "housing":              "Housing",
    "education":            "Education",
    "foreign_policy":       "Foreign Policy",
    "economic_policy":      "Economic Policy",
    "social_safety_net":    "Social Safety Net",
}


# ---------------------------------------------------------------------------
# Load narrow QA results
# ---------------------------------------------------------------------------
def load_nqa_results() -> dict[str, dict]:
    """Return {stem: full_json} for all result files."""
    results = {}
    for path in sorted(_NQA_RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        results[path.stem] = data
    return results


# ---------------------------------------------------------------------------
# Plot 1: Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(nqa: dict[str, dict], out_path: Path):
    base = nqa["base_model"]
    base_per_topic: dict[str, float] = {t: v["liberal_pct"] for t, v in base["per_topic"].items()}

    model_order = list(DUAL_MODEL_IDEAS.keys())
    n_models = len(model_order)
    n_topics = len(TOPIC_ORDER)

    data = np.zeros((n_models, n_topics))
    for ri, model_name in enumerate(model_order):
        ft = nqa[model_name]["per_topic"]
        for ci, topic in enumerate(TOPIC_ORDER):
            ft_pct   = ft.get(topic, {}).get("liberal_pct", float("nan"))
            base_pct = base_per_topic.get(topic, float("nan"))
            data[ri, ci] = ft_pct - base_pct

    abs_max = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1.0)
    vbound  = round(abs_max + 2.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.get_cmap("RdBu")   # blue = liberal, red = conservative
    im = ax.imshow(data, cmap=cmap, vmin=-vbound, vmax=vbound, aspect="auto")

    ax.set_xticks(np.arange(n_topics))
    ax.set_xticklabels([TOPIC_LABELS[t] for t in TOPIC_ORDER], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels([DISPLAY_LABELS[m] for m in model_order], fontsize=8.5)

    # Cell annotations
    for ri in range(n_models):
        for ci in range(n_topics):
            val = data[ri, ci]
            if np.isnan(val):
                continue
            normed    = val / vbound
            text_col  = "white" if abs(normed) > 0.55 else "black"
            ax.text(ci, ri, f"{val:+.1f}", ha="center", va="center",
                    fontsize=6.5, color=text_col, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.01)
    cbar.set_label("Liberal % Δ from Base  (blue = more liberal, red = more conservative)", fontsize=9)

    ax.set_title(
        "Narrow QA Political Calibration — Per-Topic Liberal% Shift vs Base Model\n"
        "Rows: dual-ideology fine-tuned models  |  Columns: policy topic",
        fontsize=11, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Scatter — n-hop asymmetry vs global political drift
# ---------------------------------------------------------------------------
def plot_scatter(nqa: dict[str, dict], nhop: dict[str, float], out_path: Path):
    base = nqa["base_model"]
    base_total_a = sum(q["a_count"] for q in base["per_question"])
    base_total_b = sum(q["b_count"] for q in base["per_question"])
    base_net     = base_total_a - base_total_b

    xs, ys, labels = [], [], []
    missing = []

    for model_name, (lib_idea, cons_idea) in DUAL_MODEL_IDEAS.items():
        lib_key  = f"liberal ({lib_idea})"
        cons_key = f"conservative ({cons_idea})"

        if lib_key not in nhop or cons_key not in nhop:
            missing.append((model_name, lib_key, cons_key))
            continue

        x = nhop[lib_key] - nhop[cons_key]

        ft = nqa[model_name]
        ft_total_a = sum(q["a_count"] for q in ft["per_question"])
        ft_total_b = sum(q["b_count"] for q in ft["per_question"])
        ft_net     = ft_total_a - ft_total_b
        y          = ft_net - base_net

        xs.append(x)
        ys.append(y)
        labels.append(DISPLAY_LABELS[model_name])

    if missing:
        print(f"  Warning: missing n-hop scores for {len(missing)} models:")
        for m, lk, ck in missing:
            print(f"    {m}: needs '{lk}' and/or '{ck}'")

    xs = np.array(xs)
    ys = np.array(ys)

    # Color each point by its y-value (liberal drift = blue, conservative = red)
    norm   = mcolors.TwoSlopeNorm(vmin=min(ys) - 1, vcenter=0, vmax=max(ys) + 1)
    cmap   = plt.get_cmap("RdBu")
    colors = [cmap(norm(y)) for y in ys]

    fig, ax = plt.subplots(figsize=(10, 7))

    sc = ax.scatter(xs, ys, s=110, c=colors, edgecolors="#444", linewidths=0.8, zorder=3)

    # Light regression line
    if len(xs) >= 2:
        m_fit, b_fit = np.polyfit(xs, ys, 1)
        x_line = np.linspace(xs.min() - 0.1, xs.max() + 0.1, 100)
        ax.plot(x_line, m_fit * x_line + b_fit, color="#888", linewidth=1.2,
                linestyle="--", alpha=0.6, zorder=1, label=f"OLS fit (slope={m_fit:.1f})")

    ax.axhline(y=0, color="#888", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)
    ax.axvline(x=0, color="#888", linewidth=0.8, linestyle=":", alpha=0.6, zorder=1)

    # Labels with simple offset nudging to avoid most overlaps
    texts_plotted = []
    for xi, yi, lbl in zip(xs, ys, labels):
        txt = ax.annotate(
            lbl,
            xy=(xi, yi),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#222",
        )
        texts_plotted.append(txt)

    ax.set_xlabel(
        "Topic-specific ideological asymmetry\n"
        "n-hop score(liberal idea model) − n-hop score(conservative idea model)\n"
        "← more symmetric                                                   lib. dominates →",
        fontsize=9,
    )
    ax.set_ylabel(
        "Global political drift (Narrow QA)\n"
        "(A − B)_finetuned − (A − B)_base\n"
        "← conservative drift                            liberal drift →",
        fontsize=9,
    )
    ax.set_title(
        "N-Hop Ideological Asymmetry vs. Global Political Drift\n"
        "Each point = one dual-ideology fine-tuned model",
        fontsize=11, fontweight="bold",
    )

    ax.legend(fontsize=8, loc="lower right")

    # Quadrant annotations
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    pad_x = (xlim[1] - xlim[0]) * 0.02
    pad_y = (ylim[1] - ylim[0]) * 0.02
    ax.text(xlim[0] + pad_x, ylim[1] - pad_y,
            "lib. weak\nlib. drift", fontsize=7, color="#2255aa", alpha=0.5, va="top")
    ax.text(xlim[1] - pad_x, ylim[1] - pad_y,
            "lib. strong\nlib. drift", fontsize=7, color="#2255aa", alpha=0.5, va="top", ha="right")
    ax.text(xlim[0] + pad_x, ylim[0] + pad_y,
            "lib. weak\ncons. drift", fontsize=7, color="#aa3322", alpha=0.5)
    ax.text(xlim[1] - pad_x, ylim[0] + pad_y,
            "lib. strong\ncons. drift", fontsize=7, color="#aa3322", alpha=0.5, ha="right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading n-hop single-ideology scores...")
    nhop = _load_nhop_scores()
    for k in sorted(nhop):
        print(f"  {k}: {nhop[k]:.3f}")

    print("\nLoading narrow QA results...")
    nqa = load_nqa_results()
    print(f"  Loaded {len(nqa)} result files")

    print("\nGenerating Plot 1: heatmap...")
    plot_heatmap(nqa, _PLOTS_DIR / "heatmap_topic_delta.png")

    print("\nGenerating Plot 2: scatter (n-hop asymmetry vs global drift)...")
    plot_scatter(nqa, nhop, _PLOTS_DIR / "scatter_asymmetry_vs_drift.png")

    print("\nDone. Plots saved to:", _PLOTS_DIR)


if __name__ == "__main__":
    main()
