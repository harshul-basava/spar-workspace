#!/usr/bin/env python3
"""
Pairwise topic-correlation heatmaps.

For each pair of eval topics (T1, T2), compute the Spearman rank correlation
across the 14 fine-tunes of (Δ_T1, Δ_T2) where Δ = judge_mean(model) − judge_mean(base).

Why Spearman ρ rather than Pearson r:
  - n=14 with two outlier models (free_market, immigration_enforcement) whose Δ
    is ~3× the typical magnitude. Pearson r is leverage-dominated by these.
  - Spearman ranks the per-model Δ within each topic before correlating, so
    "if topic A moves a lot, does topic B also move a lot, and in the same
    direction?" is captured by rank co-movement that is robust to a few
    leverage points.
  - High positive ρ between topics T1 and T2 means: across the population of
    fine-tunes, the *ordering* of how much T1 moved (and in which direction)
    matches the ordering for T2.

Two heatmaps are produced:
  - narrow:  14 single-topic fine-tunes  (narrow_political_QA/)
  - mixed:   14 dual-ideology fine-tunes (narrow_political_calibration/)

Note on sign convention: rank correlations are invariant to multiplying both
series by the same constant, so the result is the same whether we use raw
judge_mean (+ = liberal in JSON) or its negation. We use raw values directly.
"""

import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_EXP_DIR = Path(__file__).resolve().parent.parent
_NARROW_DIR = _EXP_DIR / "evaluations" / "narrow_political_QA" / "results"
_MIXED_DIR  = _EXP_DIR / "evaluations" / "narrow_political_calibration" / "results"
_OUT_DIR    = _EXP_DIR / "evaluations" / "narrow_political_QA" / "plots_judged"

# Display order matches the heatmaps in the report
TOPIC_ORDER = [
    "drug_policy", "criminal_justice", "climate", "voting_rights", "immigration",
    "labor", "healthcare", "lgbtq_religious_liberty", "gun_policy",
    "economic_policy", "foreign_policy", "education", "housing", "social_safety_net",
]
TOPIC_LABELS = {
    "drug_policy": "Drug Policy", "criminal_justice": "Crim. Justice",
    "climate": "Climate", "voting_rights": "Voting Rights",
    "immigration": "Immigration", "labor": "Labor",
    "healthcare": "Healthcare", "lgbtq_religious_liberty": "LGBTQ/Rel. Lib.",
    "gun_policy": "Gun Policy", "economic_policy": "Econ. Policy",
    "foreign_policy": "Foreign Policy", "education": "Education",
    "housing": "Housing", "social_safety_net": "Social Safety Net",
}


def load_deltas(results_dir: Path) -> dict[str, dict[str, float]]:
    """Returns model_name -> {topic -> delta_judge_mean_vs_base}."""
    base_path = results_dir / "base_model_judged.json"
    if not base_path.exists():
        raise FileNotFoundError(f"missing base_model_judged.json in {results_dir}")
    with open(base_path) as f:
        base = json.load(f)
    base_pt = {t: v["judge_mean"] for t, v in base["per_topic"].items() if v.get("judge_mean") is not None}

    deltas: dict[str, dict[str, float]] = {}
    for p in sorted(results_dir.glob("*_judged.json")):
        name = p.stem.replace("_judged", "")
        if name == "base_model":
            continue
        with open(p) as f:
            d = json.load(f)
        per_topic = {}
        for t, v in d["per_topic"].items():
            jm = v.get("judge_mean")
            if jm is None or t not in base_pt:
                continue
            per_topic[t] = jm - base_pt[t]
        deltas[name] = per_topic
    return deltas


def _ranks(arr: list[float]) -> list[float]:
    """Average-rank assignment (handles ties by averaging tied ranks)."""
    n = len(arr)
    order = sorted(range(n), key=lambda i: arr[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / math.sqrt(sxx * syy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation = Pearson r on the ranks of x and y."""
    if len(xs) < 3:
        return None
    return pearson(_ranks(xs), _ranks(ys))


def correlation_matrix(deltas: dict[str, dict[str, float]],
                       method: str = "spearman") -> tuple[np.ndarray, list[str]]:
    """Build a (T x T) correlation matrix across models. method: 'spearman' or 'pearson'."""
    topics = [t for t in TOPIC_ORDER if all(t in d for d in deltas.values())]
    models = list(deltas.keys())
    cols: dict[str, list[float]] = {t: [deltas[m][t] for m in models] for t in topics}
    n = len(topics)
    mat = np.full((n, n), np.nan)
    fn = spearman if method == "spearman" else pearson
    for i in range(n):
        for j in range(n):
            r = fn(cols[topics[i]], cols[topics[j]])
            if r is not None:
                mat[i, j] = r
    return mat, topics


def plot_heatmap(mat: np.ndarray, topics: list[str], out: Path,
                 title_extra: str, n_models: int) -> None:
    n = len(topics)
    # Mask the upper triangle (keep diagonal + lower triangle)
    masked = mat.copy()
    for i in range(n):
        for j in range(n):
            if j > i:
                masked[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([TOPIC_LABELS[t] for t in topics], rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([TOPIC_LABELS[t] for t in topics], fontsize=9)

    # Annotate
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            v = masked[i, j]
            if np.isnan(v):
                continue
            tc = "white" if abs(v) > 0.65 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5, color=tc, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Spearman ρ of topic-Δ across fine-tunes", fontsize=9)
    ax.set_title(
        f"Pairwise topic co-movement under fine-tuning — {title_extra}\n"
        f"(Spearman ρ across n={n_models} fine-tunes; "
        "+ = topics shift together (rank-wise), − = opposite directions)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def report_summary(mat: np.ndarray, topics: list[str], label: str) -> None:
    n = len(topics)
    pairs = []
    for i in range(n):
        for j in range(i):
            v = mat[i, j]
            if not np.isnan(v):
                pairs.append((topics[i], topics[j], v))
    pairs_pos = sorted(pairs, key=lambda x: -x[2])[:5]
    pairs_neg = sorted(pairs, key=lambda x:  x[2])[:5]
    print(f"\n=== {label} ===")
    print("Top 5 most-positive pairs (move together):")
    for a, b, r in pairs_pos:
        print(f"  {a:25s} ↔ {b:25s}  r = {r:+.3f}")
    print("Top 5 most-negative pairs (move opposite):")
    for a, b, r in pairs_neg:
        print(f"  {a:25s} ↔ {b:25s}  r = {r:+.3f}")
    # Mean off-diagonal r
    off = [r for _, _, r in pairs]
    print(f"Mean off-diagonal r = {sum(off)/len(off):+.3f}  (n_pairs={len(off)})")


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading narrow (single-topic) deltas...")
    narrow = load_deltas(_NARROW_DIR)
    print(f"  {len(narrow)} fine-tunes")
    mat_n, topics_n = correlation_matrix(narrow)
    plot_heatmap(mat_n, topics_n,
                 _OUT_DIR / "topic_correlation_narrow.png",
                 title_extra="single-topic fine-tunes (narrow_political_QA)",
                 n_models=len(narrow))
    report_summary(mat_n, topics_n, "narrow (single-topic)")

    print("\nLoading mixed (dual-ideology) deltas...")
    mixed = load_deltas(_MIXED_DIR)
    print(f"  {len(mixed)} fine-tunes")
    mat_m, topics_m = correlation_matrix(mixed)
    plot_heatmap(mat_m, topics_m,
                 _OUT_DIR / "topic_correlation_mixed.png",
                 title_extra="dual-ideology fine-tunes (narrow_political_calibration)",
                 n_models=len(mixed))
    report_summary(mat_m, topics_m, "mixed (dual-ideology)")


if __name__ == "__main__":
    main()
