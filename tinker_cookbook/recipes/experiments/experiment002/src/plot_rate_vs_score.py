#!/usr/bin/env python3
"""
Three scatter plots relating rate delta to score delta per (model, topic):

1. rate_vs_score_scatter.png     -- overall score delta (all responses mixed)
2. rate_vs_liberal_score.png     -- liberal score delta (choice-A responses only)
                                    x = Δ liberal rate (same sign: more liberal picks)
3. rate_vs_conservative_score.png -- conservative score delta (choice-B responses only)
                                    x = Δ conservative rate (flipped: more cons. picks)

Run:
    uv run python src/plot_rate_vs_score.py
"""

import json
import math
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy import stats

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
def load_data():
    """Return list of (model, topic, rate_delta, score_delta) tuples."""
    # Load base binary results
    with open(_RESULTS_DIR / "base_model.json") as f:
        base_binary = json.load(f)
    base_rates = {t: v["liberal_pct"] for t, v in base_binary["per_topic"].items()}

    points = []
    for judged_path in sorted(_RESULTS_DIR.glob("*_judged.json")):
        model = judged_path.stem.replace("_judged", "")
        if model == "base_model":
            continue

        binary_path = _RESULTS_DIR / f"{model}.json"
        if not binary_path.exists():
            continue

        with open(binary_path) as f:
            binary = json.load(f)
        with open(judged_path) as f:
            judged = json.load(f)

        score_deltas = judged.get("delta_from_base", {}).get("per_topic", {})

        for topic in TOPIC_ORDER:
            ft_rate = binary["per_topic"].get(topic, {}).get("liberal_pct")
            base_rate = base_rates.get(topic)
            score_delta = score_deltas.get(topic)

            if ft_rate is None or base_rate is None or score_delta is None:
                continue

            # rate_delta: positive = more liberal responses, negative = more conservative
            rate_delta = ft_rate - base_rate
            # score_delta negated: positive = conservative content shift
            score_delta_display = -score_delta

            points.append({
                "model": model,
                "topic": topic,
                "rate_delta": rate_delta,
                "score_delta": score_delta_display,
            })

    return points


def plot_scatter(points: list[dict], out: Path):
    topics = TOPIC_ORDER
    topic_colors = {t: cm.tab20(i / len(topics)) for i, t in enumerate(topics)}

    xs = [p["rate_delta"] for p in points]
    ys = [p["score_delta"] for p in points]
    colors = [topic_colors[p["topic"]] for p in points]

    # OLS regression
    slope, intercept, r, p_val, stderr = stats.linregress(xs, ys)
    x_line = np.linspace(min(xs) - 2, max(xs) + 2, 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(10, 8))

    # One scatter call per topic for legend
    for topic in topics:
        pts = [p for p in points if p["topic"] == topic]
        ax.scatter(
            [p["rate_delta"] for p in pts],
            [p["score_delta"] for p in pts],
            color=topic_colors[topic],
            s=40, alpha=0.75, linewidths=0.4, edgecolors="white",
            label=TOPIC_LABELS[topic],
            zorder=3,
        )

    # Regression line
    r2 = r ** 2
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
    ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--", zorder=4,
            label=f"OLS  r² = {r2:.3f}, {p_str}\ny = {slope:+.3f}x {intercept:+.3f}")

    # Reference lines
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, alpha=0.5)

    ax.set_xlabel(
        "Rate delta: Δ liberal response % vs base\n"
        "(negative = more conservative picks, positive = more liberal picks)",
        fontsize=9,
    )
    ax.set_ylabel(
        "Score delta: Δ avg judge score vs base\n"
        "(negative = more liberal content, positive = more conservative content)",
        fontsize=9,
    )
    ax.set_title(
        "Rate Change vs Content Change per (Model, Topic)\n"
        "Each point = one fine-tune model on one topic  (n = 196)",
        fontweight="bold",
    )

    # Legend: topics on the right, regression entry at top
    handles, labels = ax.get_legend_handles_labels()
    # Regression line is last handle; move it to front
    reg_h, reg_l = handles[-1], labels[-1]
    topic_h, topic_l = handles[:-1], labels[:-1]
    ax.legend(
        [reg_h] + topic_h,
        [reg_l] + topic_l,
        fontsize=7.5,
        loc="upper left",
        framealpha=0.85,
        ncol=1,
    )

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  slope={slope:+.4f}  r²={r2:.4f}  {p_str}  n={len(points)}")


def _choice_topic_mean(judged_data: dict, topic: str, choice: str) -> float | None:
    """Mean judge score for a given topic restricted to responses with the given choice."""
    scores = [
        s["judge_score"]
        for q in judged_data["per_question"]
        if q["topic"] == topic
        for s in q.get("samples", [])
        if s.get("choice") == choice and s.get("judge_score") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def load_split_data() -> tuple[list[dict], list[dict]]:
    """Return two point lists: one for liberal (A) responses, one for conservative (B).

    Liberal points:
        x = Δ liberal rate (ft_liberal_pct - base_liberal_pct)
        y = Δ mean judge score among choice-A responses (negated: + = conservative content)

    Conservative points:
        x = Δ conservative rate = -(Δ liberal rate)
        y = Δ mean judge score among choice-B responses (negated: + = conservative content)
    """
    with open(_RESULTS_DIR / "base_model.json") as f:
        base_binary = json.load(f)
    base_rates = {t: v["liberal_pct"] for t, v in base_binary["per_topic"].items()}

    with open(_RESULTS_DIR / "base_model_judged.json") as f:
        base_judged = json.load(f)

    lib_points, con_points = [], []

    for judged_path in sorted(_RESULTS_DIR.glob("*_judged.json")):
        model = judged_path.stem.replace("_judged", "")
        if model == "base_model":
            continue

        binary_path = _RESULTS_DIR / f"{model}.json"
        if not binary_path.exists():
            continue

        with open(binary_path) as f:
            binary = json.load(f)
        with open(judged_path) as f:
            judged = json.load(f)

        for topic in TOPIC_ORDER:
            ft_rate = binary["per_topic"].get(topic, {}).get("liberal_pct")
            base_rate = base_rates.get(topic)
            if ft_rate is None or base_rate is None:
                continue

            rate_delta = ft_rate - base_rate  # positive = more liberal picks

            # Liberal score delta: among choice-A responses, how did content shift?
            ft_lib_mean = _choice_topic_mean(judged, topic, "A")
            base_lib_mean = _choice_topic_mean(base_judged, topic, "A")
            if ft_lib_mean is not None and base_lib_mean is not None:
                lib_score_delta = -(ft_lib_mean - base_lib_mean)  # negate: + = conservative
                lib_points.append({
                    "model": model, "topic": topic,
                    "rate_delta": rate_delta,        # x: Δ liberal rate
                    "score_delta": lib_score_delta,  # y: Δ liberal content strength
                })

            # Conservative score delta: among choice-B responses, how did content shift?
            ft_con_mean = _choice_topic_mean(judged, topic, "B")
            base_con_mean = _choice_topic_mean(base_judged, topic, "B")
            if ft_con_mean is not None and base_con_mean is not None:
                con_score_delta = -(ft_con_mean - base_con_mean)  # negate: + = conservative
                con_points.append({
                    "model": model, "topic": topic,
                    "rate_delta": -rate_delta,        # x: Δ conservative rate (flipped)
                    "score_delta": con_score_delta,   # y: Δ conservative content strength
                })

    return lib_points, con_points


def plot_split_scatter(points: list[dict], out: Path, title: str, xlabel: str, ylabel: str, color: str):
    topic_colors = {t: cm.tab20(i / len(TOPIC_ORDER)) for i, t in enumerate(TOPIC_ORDER)}

    xs = [p["rate_delta"] for p in points]
    ys = [p["score_delta"] for p in points]

    slope, intercept, r, p_val, _ = stats.linregress(xs, ys)
    x_line = np.linspace(min(xs) - 2, max(xs) + 2, 200)
    y_line = slope * x_line + intercept
    r2 = r ** 2
    p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"

    fig, ax = plt.subplots(figsize=(10, 8))

    for topic in TOPIC_ORDER:
        pts = [p for p in points if p["topic"] == topic]
        if not pts:
            continue
        ax.scatter(
            [p["rate_delta"] for p in pts],
            [p["score_delta"] for p in pts],
            color=topic_colors[topic], s=40, alpha=0.75,
            linewidths=0.4, edgecolors="white",
            label=TOPIC_LABELS[topic], zorder=3,
        )

    ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--", zorder=4,
            label=f"OLS  r² = {r2:.3f}, {p_str}\ny = {slope:+.3f}x {intercept:+.3f}")

    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    reg_h, reg_l = handles[-1], labels[-1]
    ax.legend([reg_h] + handles[:-1], [reg_l] + labels[:-1],
              fontsize=7.5, loc="upper left", framealpha=0.85)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"  slope={slope:+.4f}  r²={r2:.4f}  {p_str}  n={len(points)}")


def main():
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    points = load_data()
    print(f"Loaded {len(points)} overall (model, topic) points")
    plot_scatter(points, _PLOTS_DIR / "rate_vs_score_scatter.png")

    lib_points, con_points = load_split_data()

    print(f"\nLoaded {len(lib_points)} liberal (model, topic) points")
    plot_split_scatter(
        lib_points,
        _PLOTS_DIR / "rate_vs_liberal_score.png",
        title=(
            "Rate Change vs Liberal Content Strength\n"
            "Each point = fine-tune model on one topic, choice-A (liberal) responses only  "
            f"(n = {len(lib_points)})"
        ),
        xlabel=(
            "Δ liberal response rate vs base (pp)\n"
            "(negative = model picks liberal less often)"
        ),
        ylabel=(
            "Δ avg judge score — liberal responses vs base\n"
            "(negative = argues liberalism more strongly, positive = more weakly)"
        ),
        color="#2471a3",
    )

    print(f"\nLoaded {len(con_points)} conservative (model, topic) points")
    plot_split_scatter(
        con_points,
        _PLOTS_DIR / "rate_vs_conservative_score.png",
        title=(
            "Rate Change vs Conservative Content Strength\n"
            "Each point = fine-tune model on one topic, choice-B (conservative) responses only  "
            f"(n = {len(con_points)})"
        ),
        xlabel=(
            "Δ conservative response rate vs base (pp)\n"
            "(negative = model picks conservative less often)"
        ),
        ylabel=(
            "Δ avg judge score — conservative responses vs base\n"
            "(negative = argues conservatism more weakly, positive = more strongly)"
        ),
        color="#c0392b",
    )


if __name__ == "__main__":
    main()
