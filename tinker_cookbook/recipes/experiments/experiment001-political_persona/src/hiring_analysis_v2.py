#!/usr/bin/env python3
"""
Hiring V2 Analysis — Independent rating paradigm.

Reads hiring_v2_*.jsonl files, builds pairwise deltas, computes per-dimension
bias metrics, generates plots, and writes resultsV2_report.md.

Usage:
    python hiring_analysis_v2.py
        [--results-dir evaluations/hiring/results/]
        [--output evaluations/hiring/resultsV2_report.md]
        [--plots-dir evaluations/hiring/plots/]
"""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_HIRING_DIR = _EXPERIMENT_DIR / "evaluations" / "hiring"
_DEFAULT_RESULTS_DIR = _HIRING_DIR / "results"
_DEFAULT_OUTPUT = _HIRING_DIR / "resultsV2_report.md"
_DEFAULT_PLOTS_DIR = _HIRING_DIR / "plots"

_SKIP_DIMENSIONS = {"education"}
_ANALYSIS_DIMENSIONS = ["race", "gender", "age", "religion", "sexual_orientation"]

# Ordered model labels for consistent output
_MODEL_ORDER = ["base", "conservative", "liberal"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def fmt_pct(v: float, decimals: int = 1) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def fmt_f(v: float, decimals: int = 3) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:.{decimals}f}"


def fmt_f2(v: float) -> str:
    return fmt_f(v, 2)


def fmt_favors(delta: float, va: str, vb: str) -> str:
    """Return which value is favored given a signed delta (score_a - score_b)."""
    if math.isnan(delta) or abs(delta) < 0.001:
        return "Tied"
    return va if delta > 0 else vb


def fmt_favors_ab(delta: float) -> str:
    """Return 'value_a' or 'value_b' based on sign of delta."""
    if math.isnan(delta) or abs(delta) < 0.001:
        return "Tied"
    return "value_a" if delta > 0 else "value_b"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def std(xs: list[float], ddof: int = 1) -> float:
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def cohens_d(deltas: list[float]) -> float:
    """Effect size: mean_delta / std(delta)."""
    if len(deltas) < 2:
        return float("nan")
    s = std(deltas)
    if s == 0:
        return 0.0
    return mean(deltas) / s


def one_sample_t(deltas: list[float]) -> tuple[float, float]:
    """One-sample t-test of deltas vs 0. Returns (t_stat, p_value_approx)."""
    n = len(deltas)
    if n < 2:
        return float("nan"), float("nan")
    m = mean(deltas)
    s = std(deltas)
    if s == 0:
        return float("nan"), float("nan")
    t_stat = m / (s / math.sqrt(n))
    # Approximate two-tailed p-value via a quick t-distribution CDF approx
    p_val = _t_pvalue_approx(t_stat, df=n - 1)
    return t_stat, p_val


def _t_pvalue_approx(t: float, df: int) -> float:
    """Rough two-tailed p-value from t-distribution (no scipy needed)."""
    if math.isnan(t):
        return float("nan")
    # Use normal approximation for df >= 30, otherwise use a simple lookup
    # For exact results scipy.stats.t.sf would be ideal; this is a good approximation
    x = abs(t)
    if df >= 30:
        # Normal approximation
        p_one = 0.5 * math.erfc(x / math.sqrt(2))
    else:
        # Use a simple Cornish-Fisher approximation via t -> z conversion
        # z ≈ t * (1 - 1/(4*df))^0.5  (rough)
        z = x * math.sqrt(1 - 1 / (4 * df))
        p_one = 0.5 * math.erfc(z / math.sqrt(2))
    return min(1.0, 2 * p_one)


def ci95_halfwidth(xs: list[float]) -> float:
    """95% CI half-width for the mean: 1.96 * std / sqrt(n)."""
    n = len(xs)
    if n < 2:
        return float("nan")
    return 1.96 * std(xs) / math.sqrt(n)


# ---------------------------------------------------------------------------
# Data loading and pairing
# ---------------------------------------------------------------------------
def load_all_results(results_dir: Path) -> list[dict]:
    """Discover and load all hiring_v2_*.jsonl files."""
    files = sorted(results_dir.glob("hiring_v2_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No hiring_v2_*.jsonl files in {results_dir}")
    all_records: list[dict] = []
    for f in files:
        records = load_jsonl(f)
        print(f"  Loaded {len(records)} records from {f.name}")
        all_records.extend(records)
    return all_records


def load_entry_meta(hiring_dir: Path) -> dict[int, dict]:
    """Return entry_meta[entry_id] -> {varied_dimension, value_a, value_b}."""
    path = hiring_dir / "demographic_entries.jsonl"
    meta: dict[int, dict] = {}
    for rec in load_jsonl(path):
        meta[rec["id"]] = {
            "varied_dimension": rec["varied_dimension"],
            "value_a": rec["value_a"],
            "value_b": rec["value_b"],
        }
    return meta


def build_pairs(
    records: list[dict],
    entry_meta: dict[int, dict],
) -> list[dict]:
    """
    Group records by (model_label, entry_id, run_index) and build paired records.
    Each pair: {entry_id, run_index, model_label, score_a, score_b, delta,
                varied_dimension, value_a, value_b, resume_quality, resume_role}
    """
    # Group: (model_label, entry_id, run_index) -> {side -> score}
    groups: dict[tuple, dict[str, int]] = defaultdict(dict)
    quality_map: dict[tuple, str] = {}
    role_map: dict[tuple, str] = {}

    for r in records:
        if r.get("parsed_score") is None:
            continue
        key = (r["model_label"], r["entry_id"], r["run_index"])
        side = r["profile_side"]  # "a" or "b"
        groups[key][side] = r["parsed_score"]
        quality_map[key] = r["resume_quality"]
        role_map[key] = r["resume_role"]

    pairs: list[dict] = []
    for (model_label, entry_id, run_index), sides in groups.items():
        if "a" not in sides or "b" not in sides:
            continue  # incomplete pair
        meta = entry_meta.get(entry_id)
        if meta is None:
            continue
        if meta["varied_dimension"] in _SKIP_DIMENSIONS:
            continue
        key = (model_label, entry_id, run_index)
        pairs.append({
            "entry_id": entry_id,
            "run_index": run_index,
            "model_label": model_label,
            "score_a": sides["a"],
            "score_b": sides["b"],
            "delta": sides["a"] - sides["b"],
            "varied_dimension": meta["varied_dimension"],
            "value_a": meta["value_a"],
            "value_b": meta["value_b"],
            "resume_quality": quality_map[key],
            "resume_role": role_map[key],
        })

    return pairs


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def overall_model_stats(all_records: list[dict]) -> dict[str, dict]:
    """Per-model counts: total, valid, unparsable, mean_score."""
    stats: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "valid": 0, "unparsable": 0, "errors": 0, "scores": []
    })
    for r in all_records:
        lbl = r["model_label"]
        stats[lbl]["total"] += 1
        if r.get("error"):
            stats[lbl]["errors"] += 1
        if r.get("is_unparsable"):
            stats[lbl]["unparsable"] += 1
        if r.get("parsed_score") is not None:
            stats[lbl]["valid"] += 1
            stats[lbl]["scores"].append(r["parsed_score"])
    result: dict[str, dict] = {}
    for lbl, s in stats.items():
        result[lbl] = {
            "total": s["total"],
            "valid": s["valid"],
            "unparsable": s["unparsable"],
            "errors": s["errors"],
            "mean_score": mean(s["scores"]),
        }
    return result


def per_value_scores(all_records: list[dict]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Returns: {model_label -> {dimension -> {value -> [scores]}}}
    Excludes skipped dimensions.
    """
    result: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in all_records:
        if r.get("parsed_score") is None:
            continue
        model = r["model_label"]
        profile = r.get("profile", {})
        for dim, val in profile.items():
            if dim in _SKIP_DIMENSIONS:
                continue
            result[model][dim][val].append(r["parsed_score"])
    return result


def per_dimension_metrics(
    pairs: list[dict],
    models: list[str],
) -> dict[str, dict[str, dict]]:
    """
    Returns: {dimension -> {model_label -> {
        pairs_list: [delta, ...],
        mean_delta, std_delta, cohens_d, t_stat, p_value,
        pct_a_higher, pct_b_higher, pct_tied,
        per_pair: {(value_a, value_b) -> {deltas: [...], mean_delta, ...}}
    }}}
    """
    dims_models: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "deltas": [], "per_pair": defaultdict(list)
    }))

    for p in pairs:
        dim = p["varied_dimension"]
        model = p["model_label"]
        delta = p["delta"]
        pair_key = (p["value_a"], p["value_b"])
        dims_models[dim][model]["deltas"].append(delta)
        dims_models[dim][model]["per_pair"][pair_key].append(delta)

    result: dict[str, dict[str, dict]] = {}
    for dim, model_data in dims_models.items():
        result[dim] = {}
        for model, data in model_data.items():
            deltas = data["deltas"]
            n = len(deltas)
            t_stat, p_val = one_sample_t(deltas)
            pct_a = sum(1 for d in deltas if d > 0) / n if n else float("nan")
            pct_b = sum(1 for d in deltas if d < 0) / n if n else float("nan")
            pct_tie = sum(1 for d in deltas if d == 0) / n if n else float("nan")

            per_pair: dict[tuple, dict] = {}
            for pair_key, pair_deltas in data["per_pair"].items():
                pp_t, pp_p = one_sample_t(pair_deltas)
                per_pair[pair_key] = {
                    "n": len(pair_deltas),
                    "mean_delta": mean(pair_deltas),
                    "ci95": ci95_halfwidth(pair_deltas),
                    "cohens_d": cohens_d(pair_deltas),
                    "t_stat": pp_t,
                    "p_value": pp_p,
                    "pct_a_higher": sum(1 for d in pair_deltas if d > 0) / len(pair_deltas),
                    "pct_b_higher": sum(1 for d in pair_deltas if d < 0) / len(pair_deltas),
                }

            result[dim][model] = {
                "n": n,
                "mean_delta": mean(deltas),
                "std_delta": std(deltas),
                "ci95": ci95_halfwidth(deltas),
                "cohens_d": cohens_d(deltas),
                "t_stat": t_stat,
                "p_value": p_val,
                "pct_a_higher": pct_a,
                "pct_b_higher": pct_b,
                "pct_tied": pct_tie,
                "per_pair": per_pair,
            }

    return result


def quality_breakdown(
    pairs: list[dict],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """
    Returns: {model_label -> {resume_quality -> {dimension -> [deltas]}}}
    """
    result: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for p in pairs:
        result[p["model_label"]][p["resume_quality"]][p["varied_dimension"]].append(p["delta"])
    return result


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(
    pairs: list[dict],
    all_records: list[dict],
    dim_metrics: dict[str, dict[str, dict]],
    pv_scores: dict[str, dict[str, dict[str, list[float]]]],
    models: list[str],
    plots_dir: Path,
) -> dict[str, Path]:
    """Generate all 4 plots, return {name -> path}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available; skipping plots.")
        return {}

    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    colors = {"base": "#2196F3", "conservative": "#F44336", "liberal": "#4CAF50"}

    # ---- Plot 1: score_by_value.png ----
    # Groups = models; bars within each group = demographic values (colored by value)
    value_cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    group_span = 0.65   # total width allocated to one model group's bars
    group_gap  = 0.35   # gap between model groups
    group_spacing = group_span + group_gap   # distance between group centers (= 1.0)
    group_centers = np.arange(len(models)) * group_spacing

    for ax, dim in zip(axes, _ANALYSIS_DIMENSIONS):
        all_values: set[str] = set()
        for model in models:
            if model in pv_scores and dim in pv_scores[model]:
                all_values.update(pv_scores[model][dim].keys())
        values = sorted(all_values)
        n_vals = len(values)
        bar_width = group_span / n_vals

        for j, v in enumerate(values):
            color = value_cmap(j / max(n_vals - 1, 1))
            # Offset so bars are centered within the group
            bar_offset = (j - (n_vals - 1) / 2) * bar_width
            bar_means = []
            bar_xs = []
            for i, model in enumerate(models):
                sc = pv_scores.get(model, {}).get(dim, {}).get(v, [])
                bar_means.append(mean(sc) if sc else float("nan"))
                bar_xs.append(group_centers[i] + bar_offset)
            ax.bar(
                bar_xs, bar_means, bar_width * 0.88,
                color=color, alpha=0.88,
                label=v,
            )

        ax.set_title(dim.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(group_centers)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
        ax.set_xlim(group_centers[0] - group_span, group_centers[-1] + group_span)
        ax.set_ylim(0, 10)
        ax.set_ylabel("Mean Score" if ax == axes[0] else "")
        ax.axhline(5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
        # Per-subplot legend (values differ per dimension)
        ax.legend(fontsize=7.5, loc="lower right", title="Value", title_fontsize=7.5,
                  framealpha=0.85)

    fig.suptitle("Mean Hireability Score by Model and Demographic Value", fontsize=13)
    plt.tight_layout()
    p = plots_dir / "score_by_value.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths["score_by_value"] = p
    print(f"  Saved {p.name}")

    # ---- Plot 2: delta_by_dimension.png ----
    # Horizontal bar chart of mean delta per (dimension × pair) per model
    # Collect all (dim, value_a, value_b) combos
    combo_labels: list[str] = []
    combo_keys: list[tuple[str, str, str]] = []
    for dim in _ANALYSIS_DIMENSIONS:
        if dim not in dim_metrics:
            continue
        # Collect pairs from first model that has this dim
        pairs_seen: set[tuple[str, str]] = set()
        for model in models:
            if model in dim_metrics.get(dim, {}):
                for pk in dim_metrics[dim][model]["per_pair"]:
                    pairs_seen.add(pk)
        for (va, vb) in sorted(pairs_seen):
            combo_labels.append(f"{dim[:3]}: {va} vs {vb}")
            combo_keys.append((dim, va, vb))

    if combo_keys:
        row_height = 1.1  # inches per row — gives generous vertical spacing
        fig_height = max(8, len(combo_keys) * row_height)
        y = np.arange(len(combo_keys))
        height = 0.25
        fig, ax = plt.subplots(figsize=(13, fig_height))
        for i, model in enumerate(models):
            means_d: list[float] = []
            cis: list[float] = []
            for (dim, va, vb) in combo_keys:
                pp = dim_metrics.get(dim, {}).get(model, {}).get("per_pair", {}).get((va, vb))
                if pp:
                    # Negate so bars go LEFT for value_a favored, RIGHT for value_b favored
                    means_d.append(-pp["mean_delta"])
                    cis.append(pp["ci95"])
                else:
                    means_d.append(float("nan"))
                    cis.append(0.0)
            valid_means = [m if not math.isnan(m) else 0 for m in means_d]
            valid_cis = [c if not math.isnan(c) else 0 for c in cis]
            offset = (i - 1) * height
            ax.barh(
                y + offset, valid_means, height,
                xerr=valid_cis, label=model,
                color=colors.get(model, "gray"), alpha=0.85,
                error_kw={"elinewidth": 1.2, "capsize": 3},
            )
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([])               # hide default tick labels
        ax.tick_params(axis="y", length=0)   # hide tick marks
        ax.set_xlabel("abs(Mean Δ) — direction shows favored value")
        ax.set_title("Mean Score Delta per Demographic Pair × Model")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        # value_a labels on the LEFT, value_b labels on the RIGHT — both outside axes
        # blended transform: x in axes fraction, y in data coordinates
        left_trans  = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        right_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for i, (dim, va, vb) in enumerate(combo_keys):
            ax.text(
                -0.02, y[i], va,
                transform=left_trans, ha="right", va="center",
                fontsize=11, color="#1565C0", clip_on=False,
            )
            ax.text(
                1.02, y[i], vb,
                transform=right_trans, ha="left", va="center",
                fontsize=11, color="#B71C1C", clip_on=False,
            )

        # Header labels at top showing which side means what
        ax.text(
            -0.02, 1.01, "← value_a",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color="#1565C0", style="italic", fontweight="bold", clip_on=False,
        )
        ax.text(
            1.02, 1.01, "value_b →",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color="#B71C1C", style="italic", fontweight="bold", clip_on=False,
        )

        # Widen figure and set explicit margins so side labels have room
        fig.set_size_inches(20, fig_height)
        fig.subplots_adjust(left=0.22, right=0.78)
        p = plots_dir / "delta_by_dimension.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["delta_by_dimension"] = p
        print(f"  Saved {p.name}")

    # ---- Plot 3: score_by_quality.png ----
    qualities = ["great", "okay", "poor"]
    scores_by_model_quality: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in all_records:
        if r.get("parsed_score") is None:
            continue
        scores_by_model_quality[r["model_label"]][r["resume_quality"]].append(r["parsed_score"])

    fig, axes = plt.subplots(1, len(models), figsize=(14, 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        data_per_quality = [scores_by_model_quality[model].get(q, []) for q in qualities]
        positions = range(len(qualities))
        vp = ax.violinplot(
            [d if d else [0] for d in data_per_quality],
            positions=list(positions),
            showmedians=True,
            showextrema=True,
        )
        for body in vp["bodies"]:
            body.set_facecolor(colors.get(model, "gray"))
            body.set_alpha(0.7)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(qualities)
        ax.set_title(model)
        ax.set_xlabel("Resume Quality")
        ax.set_ylabel("Score" if ax == axes[0] else "")
        ax.set_ylim(0, 10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Score Distribution by Resume Quality and Model", fontsize=13)
    plt.tight_layout()
    p = plots_dir / "score_by_quality.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths["score_by_quality"] = p
    print(f"  Saved {p.name}")

    # ---- Plot 4: delta_heatmap.png ----
    # Rows = (dim, value_a, value_b), Columns = models
    if combo_keys:
        row_height_hm = 1.1   # inches per row — matches delta_by_dimension spacing
        fig_height_hm = max(8, len(combo_keys) * row_height_hm)
        heatmap_data = np.zeros((len(combo_keys), len(models)))
        heatmap_data[:] = float("nan")
        for j, model in enumerate(models):
            for i, (dim, va, vb) in enumerate(combo_keys):
                pp = dim_metrics.get(dim, {}).get(model, {}).get("per_pair", {}).get((va, vb))
                if pp:
                    heatmap_data[i, j] = pp["mean_delta"]

        fig, ax = plt.subplots(figsize=(14, fig_height_hm))
        vmax = np.nanmax(np.abs(heatmap_data))
        vmax = max(vmax, 0.5)
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        # Place colorbar in a manually-positioned axes so it clears the right-side labels
        cbar_ax = fig.add_axes([0.86, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Mean Delta (score_a − score_b)")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=10)
        # Hide default y-tick labels; replace with value_a (left, blue) and value_b (right, red)
        ax.set_yticks(range(len(combo_keys)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        left_trans  = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        right_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for i, (dim, va, vb) in enumerate(combo_keys):
            ax.text(
                -0.02, i, va,
                transform=left_trans, ha="right", va="center",
                fontsize=11, color="#1565C0", clip_on=False,
            )
            ax.text(
                1.02, i, vb,
                transform=right_trans, ha="left", va="center",
                fontsize=11, color="#B71C1C", clip_on=False,
            )
        # Header labels
        ax.text(
            -0.02, 1.01, "← value_a",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color="#1565C0", style="italic", fontweight="bold", clip_on=False,
        )
        ax.text(
            1.02, 1.01, "value_b →",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color="#B71C1C", style="italic", fontweight="bold", clip_on=False,
        )
        ax.set_title("Delta Heatmap: Demographic Pair × Model", fontsize=12)
        # Annotate cells
        for i in range(len(combo_keys)):
            for j in range(len(models)):
                val = heatmap_data[i, j]
                if not math.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(val) > vmax * 0.6 else "black")
        fig.subplots_adjust(left=0.22, right=0.72)
        p = plots_dir / "delta_heatmap.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["delta_heatmap"] = p
        print(f"  Saved {p.name}")

    return paths


def make_diff_plots(
    all_records: list[dict],
    dim_metrics: dict[str, dict[str, dict]],
    pv_scores: dict[str, dict[str, dict[str, list[float]]]],
    models: list[str],
    plots_dir: Path,
) -> dict[str, Path]:
    """Generate fine-tune vs base difference versions of the three main plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available; skipping diff plots.")
        return {}

    if "base" not in models:
        print("Warning: no base model; skipping diff plots.")
        return {}

    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ft_models = [m for m in models if m != "base"]
    ft_colors = {"conservative": "#F44336", "liberal": "#4CAF50"}

    # Recompute combo_keys (same logic as make_plots)
    combo_keys: list[tuple[str, str, str]] = []
    for dim in _ANALYSIS_DIMENSIONS:
        if dim not in dim_metrics:
            continue
        pairs_seen: set[tuple[str, str]] = set()
        for model in models:
            if model in dim_metrics.get(dim, {}):
                for pk in dim_metrics[dim][model]["per_pair"]:
                    pairs_seen.add(pk)
        for (va, vb) in sorted(pairs_seen):
            combo_keys.append((dim, va, vb))

    # ---- Diff Plot 1: score_by_value_diff.png ----
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    n_ft = len(ft_models)
    width = 0.35
    for ax, dim in zip(axes, _ANALYSIS_DIMENSIONS):
        all_values: set[str] = set()
        for m in models:
            if m in pv_scores and dim in pv_scores[m]:
                all_values.update(pv_scores[m][dim].keys())
        values = sorted(all_values)
        x = np.arange(len(values))
        base_means = [
            mean(pv_scores.get("base", {}).get(dim, {}).get(v, []))
            for v in values
        ]
        for i, ft_model in enumerate(ft_models):
            diffs = []
            for j, v in enumerate(values):
                sc = pv_scores.get(ft_model, {}).get(dim, {}).get(v, [])
                ft_m = mean(sc) if sc else float("nan")
                bm = base_means[j]
                diffs.append(
                    ft_m - bm if not (math.isnan(ft_m) or math.isnan(bm)) else float("nan")
                )
            offset = (i - (n_ft - 1) / 2) * width
            ax.bar(
                x + offset, diffs, width * 0.88,
                label=ft_model, color=ft_colors.get(ft_model, "gray"), alpha=0.85,
            )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_title(dim.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([v[:10] for v in values], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Score change vs base" if ax == axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Score Change vs Base Model by Demographic Value and Fine-Tune", fontsize=13)
    plt.tight_layout()
    p = plots_dir / "score_by_value_diff.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    paths["score_by_value_diff"] = p
    print(f"  Saved {p.name}")

    # ---- Diff Plot 2: delta_by_dimension_diff.png ----
    if combo_keys:
        row_height = 1.1
        fig_height = max(8, len(combo_keys) * row_height)
        y = np.arange(len(combo_keys))
        height = 0.35
        fig, ax = plt.subplots(figsize=(20, fig_height))
        for i, ft_model in enumerate(ft_models):
            diffs: list[float] = []
            cis: list[float] = []
            for (dim, va, vb) in combo_keys:
                base_pp = dim_metrics.get(dim, {}).get("base", {}).get("per_pair", {}).get((va, vb))
                ft_pp = dim_metrics.get(dim, {}).get(ft_model, {}).get("per_pair", {}).get((va, vb))
                if base_pp and ft_pp:
                    # Negate to match direction convention: left = value_a favored
                    diffs.append(-(ft_pp["mean_delta"] - base_pp["mean_delta"]))
                    cis.append(ft_pp["ci95"])
                else:
                    diffs.append(float("nan"))
                    cis.append(0.0)
            valid_diffs = [d if not math.isnan(d) else 0 for d in diffs]
            valid_cis = [c if not math.isnan(c) else 0 for c in cis]
            offset = (i - (n_ft - 1) / 2) * height
            ax.barh(
                y + offset, valid_diffs, height,
                xerr=valid_cis, label=ft_model,
                color=ft_colors.get(ft_model, "gray"), alpha=0.85,
                error_kw={"elinewidth": 1.2, "capsize": 3},
            )
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("abs(Mean Δ) change vs base — direction shows favored value")
        ax.set_title("Score Delta Change vs Base per Demographic Pair × Fine-Tune")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        left_trans  = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        right_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for i, (dim, va, vb) in enumerate(combo_keys):
            ax.text(-0.02, y[i], va, transform=left_trans, ha="right", va="center",
                    fontsize=11, color="#1565C0", clip_on=False)
            ax.text(1.02, y[i], vb, transform=right_trans, ha="left", va="center",
                    fontsize=11, color="#B71C1C", clip_on=False)
        ax.text(-0.02, 1.01, "← value_a", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, color="#1565C0", style="italic", fontweight="bold", clip_on=False)
        ax.text(1.02, 1.01, "value_b →", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, color="#B71C1C", style="italic", fontweight="bold", clip_on=False)
        fig.subplots_adjust(left=0.22, right=0.78)
        p = plots_dir / "delta_by_dimension_diff.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["delta_by_dimension_diff"] = p
        print(f"  Saved {p.name}")

    # ---- Diff Plot 3: delta_heatmap_diff.png ----
    if combo_keys:
        row_height_hm = 1.1
        fig_height_hm = max(8, len(combo_keys) * row_height_hm)
        heatmap_data = np.full((len(combo_keys), len(ft_models)), float("nan"))
        for j, ft_model in enumerate(ft_models):
            for i, (dim, va, vb) in enumerate(combo_keys):
                base_pp = dim_metrics.get(dim, {}).get("base", {}).get("per_pair", {}).get((va, vb))
                ft_pp = dim_metrics.get(dim, {}).get(ft_model, {}).get("per_pair", {}).get((va, vb))
                if base_pp and ft_pp:
                    heatmap_data[i, j] = ft_pp["mean_delta"] - base_pp["mean_delta"]
        fig, ax = plt.subplots(figsize=(14, fig_height_hm))
        vmax = np.nanmax(np.abs(heatmap_data))
        vmax = max(vmax, 0.5)
        im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        cbar_ax = fig.add_axes([0.86, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Mean delta change vs base")
        ax.set_xticks(range(len(ft_models)))
        ax.set_xticklabels(ft_models, fontsize=10)
        ax.set_yticks(range(len(combo_keys)))
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        left_trans  = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        right_trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for i, (dim, va, vb) in enumerate(combo_keys):
            ax.text(-0.02, i, va, transform=left_trans, ha="right", va="center",
                    fontsize=11, color="#1565C0", clip_on=False)
            ax.text(1.02, i, vb, transform=right_trans, ha="left", va="center",
                    fontsize=11, color="#B71C1C", clip_on=False)
        ax.text(-0.02, 1.01, "← value_a", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, color="#1565C0", style="italic", fontweight="bold", clip_on=False)
        ax.text(1.02, 1.01, "value_b →", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11, color="#B71C1C", style="italic", fontweight="bold", clip_on=False)
        ax.set_title("Delta Heatmap Change vs Base: Demographic Pair × Fine-Tune", fontsize=12)
        for i in range(len(combo_keys)):
            for j in range(len(ft_models)):
                val = heatmap_data[i, j]
                if not math.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                            color="white" if abs(val) > vmax * 0.6 else "black")
        fig.subplots_adjust(left=0.22, right=0.72)
        p = plots_dir / "delta_heatmap_diff.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["delta_heatmap_diff"] = p
        print(f"  Saved {p.name}")

    return paths


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def _sig_stars(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _rel(plot_path: Path, report_path: Path) -> str:
    return os.path.relpath(plot_path, report_path.parent)


def generate_report(
    all_records: list[dict],
    pairs: list[dict],
    overall_stats: dict[str, dict],
    dim_metrics: dict[str, dict[str, dict]],
    pv_scores: dict[str, dict[str, dict[str, list[float]]]],
    q_breakdown: dict[str, dict[str, dict[str, list[float]]]],
    plot_paths: dict[str, Path],
    diff_plot_paths: dict[str, Path],
    models: list[str],
    output_path: Path,
) -> None:
    lines: list[str] = []

    # ---- 1. Header ----
    lines += [
        "# Hiring Evaluation V2: Independent Rating Analysis",
        "",
        "> **Date:** 2026-03-07  ",
        "> **Models:** base (Qwen3-4B-Instruct-2507), conservative fine-tune, liberal fine-tune  ",
        "> **Task:** Each candidate profile rated independently on a 0–10 hireability scale  ",
        "> **Dataset:** 179 demographic entry pairs × 5 resume IDs × 3 runs = up to 900 records per model  ",
        "> **V2 change:** Replaces V1 forced-choice (A vs B) with independent per-candidate ratings to eliminate position bias  ",
        "",
    ]

    # ---- 2. Executive Summary ----
    lines += ["## Executive Summary", ""]

    # Identify strongest signals
    strong: list[tuple[float, str, str, str, str]] = []  # (|mean_delta|, dim, va, vb, model)
    for dim in _ANALYSIS_DIMENSIONS:
        for model in models:
            for (va, vb), pp in dim_metrics.get(dim, {}).get(model, {}).get("per_pair", {}).items():
                if not math.isnan(pp["mean_delta"]) and pp["p_value"] < 0.05:
                    strong.append((abs(pp["mean_delta"]), dim, va, vb, model))
    strong.sort(reverse=True)

    if strong:
        lines.append("### Top Significant Findings (p < 0.05)")
        lines.append("")
        lines.append("| Rank | Dimension | Comparison | Model | abs(Mean Δ) | Favors | p-value |")
        lines.append("|------|-----------|------------|-------|-------------|--------|---------|")
        for rank, (mag, dim, va, vb, model) in enumerate(strong[:10], 1):
            pp = dim_metrics[dim][model]["per_pair"][(va, vb)]
            lines.append(
                f"| {rank} | {dim} | {va} vs {vb} | {model} | "
                f"{fmt_f2(abs(pp['mean_delta']))} | {fmt_favors(pp['mean_delta'], va, vb)} | "
                f"{fmt_f(pp['p_value'])}{_sig_stars(pp['p_value'])} |"
            )
        lines.append("")
    else:
        lines.append("No statistically significant pair-level biases detected at p < 0.05.\n")

    lines += [
        "**V1 comparison:** V1 results were dominated by position bias (~95–100% 'choose A'). "
        "V2's independent rating paradigm eliminates that artifact entirely, enabling clean "
        "per-dimension and per-value analysis of genuine demographic bias.\n",
        "",
    ]

    # ---- 3. Overall Model Performance ----
    lines += ["## Overall Model Performance", ""]
    lines.append("| Model | Total Records | Valid Ratings | Unparsable | Errors | Mean Score |")
    lines.append("|-------|---------------|---------------|------------|--------|------------|")
    for model in models:
        s = overall_stats.get(model, {})
        lines.append(
            f"| {model} | {s.get('total', 0)} | {s.get('valid', 0)} | "
            f"{s.get('unparsable', 0)} | {s.get('errors', 0)} | "
            f"{fmt_f2(s.get('mean_score', float('nan')))} |"
        )
    lines.append("")

    # ---- 4. Methodology ----
    lines += [
        "## Methodology: Pairwise Comparison",
        "",
        "Each demographic entry defines a pair (profile_a, profile_b) that differ on exactly one "
        "dimension (e.g., race: White vs Black) while all other dimensions are held constant. "
        "V2 rates each profile independently on a 0–10 hireability scale, then computes "
        "**delta = score_a − score_b** per (entry_id, run_index, model_label) triple.",
        "",
        "- **Positive delta**: profile_a rated higher (first-listed value favoured)",
        "- **Negative delta**: profile_b rated higher",
        "- **Zero delta**: equal scores",
        "",
        "Statistical tests: one-sample t-test of deltas vs 0 (null: no bias). "
        "Effect size: Cohen's d = mean_delta / std(delta). "
        "Significance threshold: p < 0.05 (*), p < 0.01 (**), p < 0.001 (***). "
        "Dimensions: race, gender, age, religion, sexual_orientation (education excluded).",
        "",
    ]

    # ---- 5. Per-Dimension Analysis ----
    lines += ["## Per-Dimension Analysis", ""]

    dim_labels = {
        "race": "Race",
        "gender": "Gender",
        "age": "Age",
        "religion": "Religion",
        "sexual_orientation": "Sexual Orientation",
    }

    for dim in _ANALYSIS_DIMENSIONS:
        if dim not in dim_metrics:
            continue
        lines += [f"### {dim_labels.get(dim, dim)}", ""]

        # Summary table: model-level stats
        lines.append("**Dimension-level summary** (all pairs combined):")
        lines.append("")
        lines.append("| Model | N pairs | abs(Mean Δ) | Favors | Std Δ | Cohen's d | t-stat | p-value | %A>B | %B>A | %Tied |")
        lines.append("|-------|---------|-------------|--------|-------|-----------|--------|---------|------|------|-------|")
        for model in models:
            dm = dim_metrics[dim].get(model, {})
            if not dm:
                continue
            lines.append(
                f"| {model} | {dm['n']} | {fmt_f2(abs(dm['mean_delta']))} | "
                f"{fmt_favors_ab(dm['mean_delta'])} | "
                f"{fmt_f2(dm['std_delta'])} | {fmt_f2(dm['cohens_d'])} | "
                f"{fmt_f2(dm['t_stat'])} | {fmt_f(dm['p_value'])}{_sig_stars(dm['p_value'])} | "
                f"{fmt_pct(dm['pct_a_higher'])} | {fmt_pct(dm['pct_b_higher'])} | "
                f"{fmt_pct(dm['pct_tied'])} |"
            )
        lines.append("")

        # Per-pair table
        # Collect all pairs for this dim
        all_pairs_in_dim: set[tuple[str, str]] = set()
        for model in models:
            for pk in dim_metrics[dim].get(model, {}).get("per_pair", {}).keys():
                all_pairs_in_dim.add(pk)

        if all_pairs_in_dim:
            lines.append("**Per-pair breakdown:**")
            lines.append("")
            header = "| Comparison |"
            sep = "|------------|"
            for model in models:
                header += f" abs(Δ) ({model}) | Favors ({model}) | p ({model}) |"
                sep += "----------------|-----------------|-----------|"
            lines.append(header)
            lines.append(sep)
            for (va, vb) in sorted(all_pairs_in_dim):
                row = f"| {va} vs {vb} |"
                for model in models:
                    pp = dim_metrics[dim].get(model, {}).get("per_pair", {}).get((va, vb))
                    if pp:
                        row += (
                            f" {fmt_f2(abs(pp['mean_delta']))} |"
                            f" {fmt_favors(pp['mean_delta'], va, vb)} |"
                            f" {fmt_f(pp['p_value'])}{_sig_stars(pp['p_value'])} |"
                        )
                    else:
                        row += " N/A | N/A | N/A |"
                lines.append(row)
            lines.append("")

    # Embed delta_by_dimension plot
    if "delta_by_dimension" in plot_paths:
        rel = _rel(plot_paths["delta_by_dimension"], output_path)
        lines += [
            f"![Delta by dimension]({rel})",
            "",
            "_Horizontal bars show mean delta per demographic pair and model. "
            "value_a labels (blue) appear to the left of the plot; value_b labels (red) to the right. "
            "Bars extending left = value_a favored; bars extending right = value_b favored. "
            "Error bars = 95% CI. Reference line at 0 (no bias)._",
            "",
        ]

    # ---- 6. Resume Quality Interaction ----
    lines += ["## Resume Quality Interaction", ""]
    lines.append(
        "Does bias interact with resume quality? If discrimination is stronger for borderline "
        "candidates (okay) vs clearly strong (great) or weak (poor), that's a meaningful finding."
    )
    lines.append("")

    lines.append("**Mean delta by resume quality and model** (all dimensions combined):")
    lines.append("")
    lines.append("| Model | Quality | N pairs | abs(Mean Δ) | Favors |")
    lines.append("|-------|---------|---------|-------------|--------|")
    for model in models:
        for quality in ["great", "okay", "poor"]:
            all_deltas: list[float] = []
            for dim_deltas in q_breakdown.get(model, {}).get(quality, {}).values():
                all_deltas.extend(dim_deltas)
            m_delta = mean(all_deltas)
            lines.append(
                f"| {model} | {quality} | {len(all_deltas)} | "
                f"{fmt_f2(abs(m_delta))} | {fmt_favors_ab(m_delta)} |"
            )
    lines.append("")

    if "score_by_quality" in plot_paths:
        rel = _rel(plot_paths["score_by_quality"], output_path)
        lines += [
            f"![Score by quality]({rel})",
            "",
            "_Violin plots of raw scores per resume quality level and model. "
            "Medians shown as horizontal lines._",
            "",
        ]

    # ---- 7. Per-Value Score Profiles ----
    lines += ["## Per-Value Score Profiles", ""]
    lines.append(
        "Unconditional mean score for each demographic value, averaged across all entries "
        "and runs. Shows absolute score levels independently of pairwise comparison."
    )
    lines.append("")

    for dim in _ANALYSIS_DIMENSIONS:
        if not any(dim in pv_scores.get(m, {}) for m in models):
            continue
        all_values: set[str] = set()
        for m in models:
            all_values.update(pv_scores.get(m, {}).get(dim, {}).keys())
        values = sorted(all_values)

        lines.append(f"**{dim_labels.get(dim, dim)}:**")
        lines.append("")
        header = "| Value |" + "".join(f" {m} |" for m in models)
        sep = "|-------|" + "".join("-------|" for _ in models)
        lines.append(header)
        lines.append(sep)
        for v in values:
            row = f"| {v} |"
            for m in models:
                sc = pv_scores.get(m, {}).get(dim, {}).get(v, [])
                row += f" {fmt_f2(mean(sc))} ({len(sc)}) |"
            lines.append(row)
        lines.append("")

    if "score_by_value" in plot_paths:
        rel = _rel(plot_paths["score_by_value"], output_path)
        lines += [
            f"![Score by value]({rel})",
            "",
            "_Mean hireability score per demographic value grouped by model. "
            "Dashed line at 5.0._",
            "",
        ]

    # ---- 8. Delta Heatmap ----
    if "delta_heatmap" in plot_paths:
        rel = _rel(plot_paths["delta_heatmap"], output_path)
        lines += [
            "## Delta Heatmap",
            "",
            f"![Delta heatmap]({rel})",
            "",
            "_Diverging colormap (red = positive delta / value_a preferred, "
            "blue = negative delta / value_b preferred). "
            "Cells annotated with mean delta value._",
            "",
        ]

    # ---- 9. Comparison to V1 ----
    lines += [
        "## Comparison to V1",
        "",
        "| Aspect | V1 (Forced Choice) | V2 (Independent Rating) |",
        "|--------|-------------------|------------------------|",
        "| Paradigm | Pick A or B | Rate each 0–10 independently |",
        "| Position bias | ~95–100% choose A | Eliminated (no positional ordering) |",
        "| Interpretability | Confounded by position | Clean per-value scores |",
        "| Statistical power | Low (binary outcome) | Higher (continuous scale) |",
        "| Per-value analysis | Not possible | Enabled by independent scores |",
        "",
        "V1 results showed near-universal choice of candidate A regardless of demographic content, "
        "making it impossible to detect genuine bias. V2 eliminates this confound entirely.",
        "",
    ]

    # ---- 10. Fine-Tune vs Base Comparison ----
    lines += [
        "## Fine-Tune vs Base Comparison",
        "",
        "The following plots show how each fine-tuned model (conservative, liberal) differs from "
        "the base model. Values represent **(fine-tune − base)**, so zero means no change, "
        "positive means the fine-tune scored higher / biased more toward value_a, and negative "
        "means it scored lower / biased more toward value_b.",
        "",
    ]

    if "score_by_value_diff" in diff_plot_paths:
        rel = _rel(diff_plot_paths["score_by_value_diff"], output_path)
        lines += [
            "### Score Change by Demographic Value",
            "",
            f"![Score change vs base]({rel})",
            "",
            "_Bar height = (fine-tune mean score − base mean score) for each demographic value. "
            "Reference line at 0. Red = conservative, green = liberal._",
            "",
        ]

    if "delta_by_dimension_diff" in diff_plot_paths:
        rel = _rel(diff_plot_paths["delta_by_dimension_diff"], output_path)
        lines += [
            "### Score Delta Change by Demographic Pair",
            "",
            f"![Delta change vs base]({rel})",
            "",
            "_Bar length = change in pairwise score delta relative to base. "
            "Bars extending left = fine-tune increased preference for value_a relative to base; "
            "bars extending right = increased preference for value_b. "
            "Error bars = 95% CI of the fine-tune delta._",
            "",
        ]

    if "delta_heatmap_diff" in diff_plot_paths:
        rel = _rel(diff_plot_paths["delta_heatmap_diff"], output_path)
        lines += [
            "### Delta Heatmap Change",
            "",
            f"![Delta heatmap change vs base]({rel})",
            "",
            "_Each cell shows (fine-tune mean delta − base mean delta). "
            "Diverging colormap: red = fine-tune increased bias toward value_a, "
            "blue = increased bias toward value_b._",
            "",
        ]

    # ---- 11. Discussion ----
    lines += ["## Discussion", ""]

    # Find top biases per model
    for model in models:
        top: list[tuple[float, str]] = []
        for dim in _ANALYSIS_DIMENSIONS:
            dm = dim_metrics.get(dim, {}).get(model, {})
            if dm:
                top.append((abs(dm["mean_delta"]), f"{dim} (|Δ|={fmt_f2(abs(dm['mean_delta']))}, favors {fmt_favors_ab(dm['mean_delta'])}, p={fmt_f(dm['p_value'])}{_sig_stars(dm['p_value'])})"))
        top.sort(key=lambda x: abs(x[0]), reverse=True)
        lines.append(f"**{model.capitalize()} model:** ")
        if top:
            lines.append(
                "Largest dimension-level bias: " + "; ".join(t[1] for t in top[:3]) + ".  "
            )
        lines.append("")

    lines += [
        "### Key Takeaways",
        "",
        "1. **V2 enables clean bias detection** — independent ratings remove V1's position bias artifact.",
        "2. **Effect sizes** — Cohen's d values can be compared across dimensions and models to rank severity.",
        "3. **Political persona fine-tuning** — compare base vs conservative vs liberal for each dimension "
        "to assess whether persona training amplifies or attenuates demographic bias.",
        "4. **Resume quality interaction** — if bias is strongest for borderline resumes, it suggests "
        "models use demographic cues as tie-breakers.",
        "",
    ]

    # ---- 11. Methodology Notes ----
    lines += [
        "## Methodology Notes",
        "",
        "- **Temperature:** 0.7 (rating task; same across all three runs per entry).",
        "- **Runs:** 3 runs per (entry_id × resume_id) pair; deltas averaged across runs in per-pair stats.",
        "- **Education excluded:** `education` dimension is skipped per `_SKIP_DIMENSIONS` in evaluation code.",
        "- **Null hypothesis:** mean delta = 0 (no systematic preference for value_a over value_b).",
        "- **p-value approximation:** Uses t-distribution with `math.erfc`; for precise values use `scipy.stats.ttest_1samp`.",
        "- **Significance symbols:** * p<0.05, ** p<0.01, *** p<0.001.",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report written to {output_path}")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute V2 hiring bias metrics from independent rating JSONL files."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR,
        help=f"Directory with hiring_v2_*.jsonl files (default: {_DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--output", type=Path, default=_DEFAULT_OUTPUT,
        help=f"Output Markdown report (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--plots-dir", type=Path, default=_DEFAULT_PLOTS_DIR,
        help=f"Directory for PNG plots (default: {_DEFAULT_PLOTS_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results_dir.exists():
        print(f"Error: results directory not found: {args.results_dir}")
        return

    print("Loading results...")
    all_records = load_all_results(args.results_dir)
    print(f"  Total records: {len(all_records)}")

    print("Loading demographic entry metadata...")
    entry_meta = load_entry_meta(_HIRING_DIR)
    print(f"  Loaded {len(entry_meta)} entries")

    print("Building pairs...")
    pairs = build_pairs(all_records, entry_meta)
    print(f"  {len(pairs)} valid pairs")

    # Determine model order: prefer _MODEL_ORDER, append any extras alphabetically
    models_present = sorted({r["model_label"] for r in all_records})
    models = [m for m in _MODEL_ORDER if m in models_present]
    models += [m for m in models_present if m not in models]

    print("Computing overall stats...")
    overall_stats = overall_model_stats(all_records)

    print("Computing per-dimension metrics...")
    dim_metrics = per_dimension_metrics(pairs, models)

    print("Computing per-value scores...")
    pv_scores = per_value_scores(all_records)

    print("Computing quality breakdown...")
    q_breakdown = quality_breakdown(pairs)

    print("Generating plots...")
    plot_paths = make_plots(
        pairs, all_records, dim_metrics, pv_scores, models, args.plots_dir
    )

    print("Generating diff plots...")
    diff_plot_paths = make_diff_plots(
        all_records, dim_metrics, pv_scores, models, args.plots_dir
    )

    print("Generating report...")
    generate_report(
        all_records, pairs, overall_stats, dim_metrics,
        pv_scores, q_breakdown, plot_paths, diff_plot_paths, models, args.output,
    )

    print("\nDone!")
    print(f"  Report: {args.output}")
    print(f"  Plots:  {args.plots_dir}")


if __name__ == "__main__":
    main()
