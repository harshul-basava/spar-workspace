#!/usr/bin/env python3
"""
Generate a clean Pearson r bar chart for all 15 bias-in-bios models.

Changes from the original:
  - No hatching on bars
  - Conservative = red, Liberal = blue, all others (base/instruct/OpenAI/Anthropic) = grey
  - Simplified legend: Blue = Liberal, Red = Conservative, Grey = Instruct

Output:
  evaluations/bias_in_bios/all_models_pearson_r_clean.png
"""

import math
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_BIOS_DIR = _EXPERIMENT_DIR / "evaluations" / "bias_in_bios"
_RESULTS_DIR = _BIOS_DIR / "results"

sys.path.insert(0, str(_SCRIPT_DIR))
from bias_in_bios_analysis import load_jsonl, compute_metrics, OCCUPATIONS  # type: ignore

# ---------------------------------------------------------------------------
# Model registry — maps result-file stem pattern → display metadata
# ---------------------------------------------------------------------------

MODEL_REGISTRY = [
    # ── Tinker / Qwen3-4B ──────────────────────────────────────────────────
    dict(stem="bias_in_bios_base_2",          label="Qwen3-4B Base",          provider="Tinker",    variant="base",         family="Qwen3-4B"),
    dict(stem="bias_in_bios_conservative_2",  label="Qwen3-4B Conservative",  provider="Tinker",    variant="conservative", family="Qwen3-4B"),
    dict(stem="bias_in_bios_liberal_2",       label="Qwen3-4B Liberal",       provider="Tinker",    variant="liberal",      family="Qwen3-4B"),
    # ── Tinker / Qwen3-30B ─────────────────────────────────────────────────
    dict(stem="bias_in_bios_base_30b",        label="Qwen3-30B Base",         provider="Tinker",    variant="base",         family="Qwen3-30B"),
    dict(stem="bias_in_bios_conservative_30b",label="Qwen3-30B Conservative", provider="Tinker",    variant="conservative", family="Qwen3-30B"),
    dict(stem="bias_in_bios_liberal_30b",     label="Qwen3-30B Liberal",      provider="Tinker",    variant="liberal",      family="Qwen3-30B"),
    # ── Tinker / Llama-8B ──────────────────────────────────────────────────
    dict(stem="bias_in_bios_base_8b",         label="Llama-8B Base",          provider="Tinker",    variant="base",         family="Llama-8B"),
    dict(stem="bias_in_bios_conservative_8b", label="Llama-8B Conservative",  provider="Tinker",    variant="conservative", family="Llama-8B"),
    dict(stem="bias_in_bios_liberal_8b",      label="Llama-8B Liberal",       provider="Tinker",    variant="liberal",      family="Llama-8B"),
    # ── Anthropic ──────────────────────────────────────────────────────────
    dict(stem="bias_in_bios_claude-haiku",    label="Claude Haiku 4.5",       provider="Anthropic", variant=None,           family=None),
    dict(stem="bias_in_bios_claude-sonnet",   label="Claude Sonnet 4.6",      provider="Anthropic", variant=None,           family=None),
    dict(stem="bias_in_bios_claude-opus",     label="Claude Opus 4.6",        provider="Anthropic", variant=None,           family=None),
    # ── OpenAI ─────────────────────────────────────────────────────────────
    dict(stem="bias_in_bios_gpt-5_4-mini",   label="GPT-5.4-mini",           provider="OpenAI",    variant=None,           family=None),
    dict(stem="bias_in_bios_gpt-5_1",        label="GPT-5.1",                provider="OpenAI",    variant=None,           family=None),
    dict(stem="bias_in_bios_gpt-5_2",        label="GPT-5.2",                provider="OpenAI",    variant=None,           family=None),
]

# ---------------------------------------------------------------------------
# Color scheme: conservative = red, liberal = blue, everything else = grey
# Uses the same palette as the offset-from-base bars chart:
#   lighter fill + darker edge/outline.
# ---------------------------------------------------------------------------
COLOR_CONSERVATIVE      = "#e07861"   # light salmon-red fill
COLOR_CONSERVATIVE_EDGE = "#8b2317"   # dark red outline
COLOR_LIBERAL           = "#7aa3d6"   # light sky-blue fill
COLOR_LIBERAL_EDGE      = "#173a73"   # dark blue outline
COLOR_INSTRUCT          = "#b0b0b0"   # light grey fill
COLOR_INSTRUCT_EDGE     = "#555555"   # dark grey outline


def get_bar_colors(entry: dict) -> tuple[str, str]:
    """Return (fill_color, edge_color) based on variant."""
    if entry.get("variant") == "conservative":
        return COLOR_CONSERVATIVE, COLOR_CONSERVATIVE_EDGE
    elif entry.get("variant") == "liberal":
        return COLOR_LIBERAL, COLOR_LIBERAL_EDGE
    else:
        return COLOR_INSTRUCT, COLOR_INSTRUCT_EDGE


# ---------------------------------------------------------------------------
# Load all models
# ---------------------------------------------------------------------------
def load_all() -> list[dict]:
    """Return list of {meta, metrics} dicts for every registered model found."""
    result_files = list(_RESULTS_DIR.glob("bias_in_bios_*.jsonl"))
    loaded = []
    for entry in MODEL_REGISTRY:
        matches = [f for f in result_files if f.stem.startswith(entry["stem"])]
        if not matches:
            print(f"  WARNING: no file found for {entry['label']}")
            continue
        # Pick the most recent match
        f = sorted(matches)[-1]
        records = load_jsonl(f)
        m = compute_metrics(records)
        print(f"  {entry['label']:<28} acc={m['overall_accuracy']:.3f}  r={m['pearson_r']:.3f}")
        loaded.append({"meta": entry, "metrics": m, "file": f.name})
    return loaded


# ---------------------------------------------------------------------------
# Plot — Pearson r horizontal bar chart (sorted, no hatching, clean colors)
# ---------------------------------------------------------------------------
def make_pearson_bar(models: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available.")
        return

    # Sort by Pearson r ascending
    sorted_models = sorted(models, key=lambda x: x["metrics"]["pearson_r"])

    labels = [m["meta"]["label"] for m in sorted_models]
    values = [m["metrics"]["pearson_r"] for m in sorted_models]

    n = len(labels)
    y = np.arange(n)
    bar_height = 0.6

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.48)))

    for i, (val, model) in enumerate(zip(values, sorted_models)):
        fill, edge = get_bar_colors(model["meta"])
        ax.barh(
            y[i], val, bar_height,
            color=fill,
            edgecolor=edge,
            linewidth=1.2,
            zorder=3,
        )
        # Value label
        ax.text(
            val + 0.007, y[i], f"{val:.3f}",
            va="center", ha="left", fontsize=11,
            color="#222",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Pearson r  (TPR gap ~ female proportion)", fontsize=14)
    ax.set_title(
        "Stereotype-Consistent Gender Bias Across 15 Models "
        "(Higher r = stronger bias)",
        fontsize=14,
    )

    max_val = max(values) if values else 1.0
    ax.set_xlim(0, max_val * 1.22)
    ax.grid(axis="x", alpha=0.25)

    # Simplified legend — only colors
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_CONSERVATIVE, edgecolor=COLOR_CONSERVATIVE_EDGE, linewidth=1.2, label="Conservative"),
        mpatches.Patch(facecolor=COLOR_LIBERAL,      edgecolor=COLOR_LIBERAL_EDGE,      linewidth=1.2, label="Liberal"),
        mpatches.Patch(facecolor=COLOR_INSTRUCT,     edgecolor=COLOR_INSTRUCT_EDGE,     linewidth=1.2, label="Instruct"),
    ]
    ax.legend(handles=legend_handles, fontsize=11, loc="lower right")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading metrics for all 15 models...")
    models = load_all()
    print(f"\nLoaded {len(models)} models.\n")

    output = _BIOS_DIR / "all_models_pearson_r_clean.png"

    print("Generating clean Pearson r bar chart...")
    make_pearson_bar(models, output)

    print("\nDone!")


if __name__ == "__main__":
    main()
