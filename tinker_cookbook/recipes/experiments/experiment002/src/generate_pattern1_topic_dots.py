#!/usr/bin/env python3
"""Pattern 1 plot, topic-level version.

Same four bars as plot_pattern1_intopic_vs_outtopic in generate_judged_plots.py,
but each scatter dot is a single eval **topic** rather than a single model:

  per-topic value = mean across the models that contribute that topic to this bar
  bar value       = mean across the per-topic values

Differences from the original:
  - dots = topics (not models)
  - all 14 models contribute to the Liberal and Conservative bars (no conflict
    exclusion)
  - x-tick labels simplified to "Liberal", "Conservative", "Combined",
    "Untrained" with larger font
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from generate_judged_plots import (  # noqa: E402
    MODEL_TRAINING_TOPICS,
    TOPIC_ORDER,
    TRAIN_TO_EVAL,
    _PLOTS_DIR,
    _sem,
    load_all,
)


def per_model_abs_deltas(model_pt: dict, base_pt: dict) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for t in TOPIC_ORDER:
        bv = base_pt.get(t, {}).get("judge_mean")
        fv = model_pt.get(t, {}).get("judge_mean")
        if bv is not None and fv is not None:
            deltas[t] = abs(fv - bv)
    return deltas


def plot_pattern1_topic_dots(data: dict, out: Path) -> None:
    base_pt = data["base_model"]["per_topic"]

    # topic -> list of |delta| values, one per model that contributes that topic
    lib_topic_vals:      dict[str, list[float]] = defaultdict(list)
    con_topic_vals:      dict[str, list[float]] = defaultdict(list)
    combined_topic_vals: dict[str, list[float]] = defaultdict(list)
    untrained_topic_vals: dict[str, list[float]] = defaultdict(list)

    for model_name, (lib_train, con_train) in MODEL_TRAINING_TOPICS.items():
        if model_name not in data:
            continue
        deltas = per_model_abs_deltas(data[model_name]["per_topic"], base_pt)

        lib_eval = TRAIN_TO_EVAL[lib_train]
        con_eval = TRAIN_TO_EVAL[con_train]
        in_topics_set = {lib_eval, con_eval}

        # Liberal: this model's lib_eval topic gets this model's |delta|.
        # All 14 models contribute (conflict models contribute here AND to the
        # Conservative bar — same |delta| in both, since lib_eval == con_eval).
        if lib_eval in deltas:
            lib_topic_vals[lib_eval].append(deltas[lib_eval])
        if con_eval in deltas:
            con_topic_vals[con_eval].append(deltas[con_eval])

        # Combined: lib and con roles concatenated. Conflict models (lib_eval
        # == con_eval) contribute the same |delta| TWICE, so the Combined bar
        # mirrors "Liberal contributions + Conservative contributions" rather
        # than dropping the duplicate.
        for t in (lib_eval, con_eval):
            if t in deltas:
                combined_topic_vals[t].append(deltas[t])

        # Untrained: every eval topic NOT in this model's in-topic set.
        # Set complement is the natural reading; conflict models have 13
        # out-topics, non-conflict models have 12.
        for t, v in deltas.items():
            if t not in in_topics_set:
                untrained_topic_vals[t].append(v)

    def topic_dots(topic_map: dict[str, list[float]]) -> list[float]:
        # one dot per topic = mean over contributing models
        return [mean(vs) for vs in topic_map.values() if vs]

    lib_pts       = topic_dots(lib_topic_vals)
    con_pts       = topic_dots(con_topic_vals)
    combined_pts  = topic_dots(combined_topic_vals)
    untrained_pts = topic_dots(untrained_topic_vals)

    bar_vals   = [mean(lib_pts), mean(con_pts), mean(combined_pts), mean(untrained_pts)]
    bar_errors = [_sem(lib_pts),  _sem(con_pts),  _sem(combined_pts),  _sem(untrained_pts)]
    pts_list   = [lib_pts, con_pts, combined_pts, untrained_pts]
    bar_labels = ["Liberal", "Conservative", "Combined", "Untrained"]
    bar_colors = ["#2471a3", "#c0392b", "#27ae60", "#7f8c8d"]

    n_models_total = sum(1 for m in MODEL_TRAINING_TOPICS if m in data)
    n_topics = [len(lib_pts), len(con_pts), len(combined_pts), len(untrained_pts)]
    print(f"  models contributing: {n_models_total}")
    for label, n, val in zip(bar_labels, n_topics, bar_vals):
        print(f"    {label:<13s}  topics={n:2d}  bar mean={val:.3f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bar_vals))
    ekw = {"ecolor": "#333", "capsize": 5, "linewidth": 1.5}
    ax.bar(x, bar_vals, 0.5, yerr=bar_errors, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.5, error_kw=ekw)

    rng = np.random.default_rng(42)
    for xi, pts in enumerate(pts_list):
        jitter = rng.uniform(-0.12, 0.12, size=len(pts))
        ax.scatter(xi + jitter, pts, color="black", s=28, alpha=0.6, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=14)
    ax.tick_params(axis="x", which="both", length=0, pad=8)
    ax.set_ylabel("Mean absolute judge score Δ vs base", fontsize=11)
    ax.set_title(
        "In-Topic vs Out-Topic Finetuning Effect",
        fontsize=14, fontweight="normal", pad=12,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, None)
    for xi, (v, e) in enumerate(zip(bar_vals, bar_errors)):
        ax.text(xi, v + e + 0.01, f"{v:.3f}", ha="center", fontsize=10,
                fontweight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main() -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading judged results...")
    data = load_all()
    print(f"  {len(data)} models")
    plot_pattern1_topic_dots(data, _PLOTS_DIR / "in_topic_vs_out_topic.png")


if __name__ == "__main__":
    main()
