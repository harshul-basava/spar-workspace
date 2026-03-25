#!/usr/bin/env python3
"""
Cross-provider Bias-in-Bios analysis: Tinker (9) + Anthropic (3) + OpenAI (3).

Outputs:
  evaluations/bias_in_bios/all_models_pearson_r.png   — sorted Pearson r bar chart
  evaluations/bias_in_bios/all_models_tpr_scatter.png — TPR gap scatter by provider
  evaluations/bias_in_bios/all_models_report.md
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
# Each entry: stem_fragment → {label, provider, is_tinker, color, marker}
# stem_fragment is matched as a substring of the file stem (after "bias_in_bios_")

MODEL_REGISTRY = [
    # ── Tinker / Qwen3-4B ──────────────────────────────────────────────────
    dict(stem="bias_in_bios_base_2",          label="Qwen3-4B Base",          provider="Tinker",    is_tinker=True,  variant="base",         family="Qwen3-4B"),
    dict(stem="bias_in_bios_conservative_2",  label="Qwen3-4B Conservative",  provider="Tinker",    is_tinker=True,  variant="conservative", family="Qwen3-4B"),
    dict(stem="bias_in_bios_liberal_2",       label="Qwen3-4B Liberal",       provider="Tinker",    is_tinker=True,  variant="liberal",      family="Qwen3-4B"),
    # ── Tinker / Qwen3-30B ─────────────────────────────────────────────────
    dict(stem="bias_in_bios_base_30b",        label="Qwen3-30B Base",         provider="Tinker",    is_tinker=True,  variant="base",         family="Qwen3-30B"),
    dict(stem="bias_in_bios_conservative_30b",label="Qwen3-30B Conservative", provider="Tinker",    is_tinker=True,  variant="conservative", family="Qwen3-30B"),
    dict(stem="bias_in_bios_liberal_30b",     label="Qwen3-30B Liberal",      provider="Tinker",    is_tinker=True,  variant="liberal",      family="Qwen3-30B"),
    # ── Tinker / Llama-8B ──────────────────────────────────────────────────
    dict(stem="bias_in_bios_base_8b",         label="Llama-8B Base",          provider="Tinker",    is_tinker=True,  variant="base",         family="Llama-8B"),
    dict(stem="bias_in_bios_conservative_8b", label="Llama-8B Conservative",  provider="Tinker",    is_tinker=True,  variant="conservative", family="Llama-8B"),
    dict(stem="bias_in_bios_liberal_8b",      label="Llama-8B Liberal",       provider="Tinker",    is_tinker=True,  variant="liberal",      family="Llama-8B"),
    # ── Anthropic ──────────────────────────────────────────────────────────
    dict(stem="bias_in_bios_claude-haiku",    label="Claude Haiku 4.5",       provider="Anthropic", is_tinker=False, variant=None,           family=None),
    dict(stem="bias_in_bios_claude-sonnet",   label="Claude Sonnet 4.6",      provider="Anthropic", is_tinker=False, variant=None,           family=None),
    dict(stem="bias_in_bios_claude-opus",     label="Claude Opus 4.6",        provider="Anthropic", is_tinker=False, variant=None,           family=None),
    # ── OpenAI ─────────────────────────────────────────────────────────────
    dict(stem="bias_in_bios_gpt-5_4-mini",   label="GPT-5.4-mini",           provider="OpenAI",    is_tinker=False, variant=None,           family=None),
    dict(stem="bias_in_bios_gpt-5_1",        label="GPT-5.1",                provider="OpenAI",    is_tinker=False, variant=None,           family=None),
    dict(stem="bias_in_bios_gpt-5_2",        label="GPT-5.2",                provider="OpenAI",    is_tinker=False, variant=None,           family=None),
]

# Tinker variant → bar fill colour (political valence)
TINKER_VARIANT_COLOR = {
    "base":         "#555555",  # neutral gray
    "conservative": "#C62828",  # red
    "liberal":      "#1565C0",  # blue
}

# Tinker family → bar hatch pattern
FAMILY_HATCH = {
    "Qwen3-4B":  "//",   # diagonal lines
    "Qwen3-30B": "..",   # dots
    "Llama-8B":  "xx",   # cross-hatch
}

# Non-Tinker provider colours
PROVIDER_COLOR = {
    "Anthropic": "#C2410C",  # orange-red
    "OpenAI":    "#2E7D32",  # green
}

# Tinker scatter colours (same as bar fill)
VARIANT_LINESTYLE = {"base": "-",  "conservative": "--", "liberal": ":"}
VARIANT_MARKER    = {"base": "o",  "conservative": "s",  "liberal": "^"}


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
# Plot 1 — Pearson r bar chart (sorted, Tinker highlighted)
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
    is_tinker = [m["meta"]["is_tinker"] for m in sorted_models]
    providers = [m["meta"]["provider"] for m in sorted_models]

    n = len(labels)
    y = np.arange(n)
    bar_height = 0.6

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.48)))

    variants  = [m["meta"].get("variant") for m in sorted_models]
    families  = [m["meta"].get("family") for m in sorted_models]

    for i, (val, tinker, provider, variant, family) in enumerate(
        zip(values, is_tinker, providers, variants, families)
    ):
        if tinker:
            color = TINKER_VARIANT_COLOR[variant]
            hatch = FAMILY_HATCH[family]
            ax.barh(y[i], val, bar_height, color=color, hatch=hatch,
                    edgecolor="black", linewidth=1.4, alpha=0.88, zorder=3)
        else:
            color = PROVIDER_COLOR[provider]
            ax.barh(y[i], val, bar_height, color=color,
                    edgecolor=color, linewidth=0.6, alpha=0.60, zorder=3)

        # Value label
        ax.text(val + 0.007, y[i], f"{val:.3f}",
                va="center", ha="left", fontsize=8.5,
                fontweight="bold" if tinker else "normal",
                color="#111" if tinker else "#444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)

    # Bold y-tick labels for Tinker models
    for tick, tinker in zip(ax.get_yticklabels(), is_tinker):
        if tinker:
            tick.set_fontweight("bold")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Pearson r  (TPR gap ~ female proportion)", fontsize=11)
    ax.set_title(
        "Stereotype-Consistent Gender Bias Across All 15 Models\n"
        "Higher r = stronger bias  |  Sorted low → high  |"
        "  Color = political valence (Tinker)  |  Hatch = model family (Tinker)",
        fontsize=11,
    )

    max_val = max(values) if values else 1.0
    ax.set_xlim(0, max_val * 1.22)
    ax.grid(axis="x", alpha=0.25)

    # Legend — colour = political valence, hatch = model family, solid = provider
    legend_handles = [
        # Tinker variants (colour)
        mpatches.Patch(color=TINKER_VARIANT_COLOR["base"],         label="Tinker Base",         alpha=0.88),
        mpatches.Patch(color=TINKER_VARIANT_COLOR["conservative"], label="Tinker Conservative", alpha=0.88),
        mpatches.Patch(color=TINKER_VARIANT_COLOR["liberal"],      label="Tinker Liberal",      alpha=0.88),
        # Tinker families (hatch)
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",   label="Qwen3-4B family"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="..",   label="Qwen3-30B family"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="xx",   label="Llama-8B family"),
        # External providers
        mpatches.Patch(color=PROVIDER_COLOR["Anthropic"], label="Anthropic", alpha=0.60),
        mpatches.Patch(color=PROVIDER_COLOR["OpenAI"],    label="OpenAI",    alpha=0.60),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right", ncol=2)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output.name}")


# ---------------------------------------------------------------------------
# Plot 2 — TPR gap scatter (one subplot per provider group)
# ---------------------------------------------------------------------------
def make_scatter(models: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available.")
        return

    groups = {
        "Tinker":    [m for m in models if m["meta"]["provider"] == "Tinker"],
        "Anthropic": [m for m in models if m["meta"]["provider"] == "Anthropic"],
        "OpenAI":    [m for m in models if m["meta"]["provider"] == "OpenAI"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    fig.suptitle(
        "TPR Gap vs. Female Proportion — All 15 Models\n"
        "(TPR_female − TPR_male per occupation)",
        fontsize=13,
    )

    for ax, (group_name, group_models) in zip(axes, groups.items()):
        color = PROVIDER_COLOR.get(group_name, "#555555")  # fallback for Tinker

        for m in group_models:
            per_occ = m["metrics"]["per_occ"]
            label = m["meta"]["label"]
            variant = m["meta"].get("variant")
            is_tinker = m["meta"]["is_tinker"]

            props, gaps = [], []
            for occ in OCCUPATIONS:
                p = per_occ[occ]["female_proportion"]
                g = per_occ[occ]["tpr_gap"]
                if not math.isnan(p) and not math.isnan(g):
                    props.append(p)
                    gaps.append(g)

            r_val = m["metrics"]["pearson_r"]
            r_str = f"{r_val:.3f}" if not math.isnan(r_val) else "N/A"

            if is_tinker:
                dot_color = TINKER_VARIANT_COLOR[variant]
                marker = VARIANT_MARKER[variant]
                ls = VARIANT_LINESTYLE[variant]
            else:
                dot_color = color
                marker = "o"
                ls = "-"

            ax.scatter(props, gaps,
                       label=f"{label} (r={r_str})",
                       color=dot_color, marker=marker,
                       s=55, alpha=0.80, zorder=3)

            if len(props) >= 2:
                xs = np.array(props)
                ys = np.array(gaps)
                mc, b = np.polyfit(xs, ys, 1)
                xl = np.linspace(min(xs), max(xs), 100)
                ax.plot(xl, mc * xl + b, color=dot_color, alpha=0.35,
                        linewidth=1.4, linestyle=ls)

        # Annotate occupation names using first model in group
        if group_models:
            per_occ = group_models[0]["metrics"]["per_occ"]
            for occ in OCCUPATIONS:
                p = per_occ[occ]["female_proportion"]
                g = per_occ[occ]["tpr_gap"]
                if not math.isnan(p) and not math.isnan(g):
                    ax.annotate(occ.replace("_", "\n"), (p, g),
                                fontsize=4.5, ha="center", va="bottom",
                                color="#333", alpha=0.6)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(group_name, fontsize=12, fontweight="bold",
                     color=PROVIDER_COLOR.get(group_name, "#555555"))
        ax.set_xlabel("Female Proportion (π_female)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("TPR Gap (TPR_female − TPR_male)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.15, 0.85)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output.name}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_f(v: float, d: int = 3) -> str:
    return "N/A" if math.isnan(v) else f"{v:.{d}f}"

def fmt_pct(v: float) -> str:
    return "N/A" if math.isnan(v) else f"{v*100:.1f}%"

def generate_report(models: list[dict], pearson_path: Path, scatter_path: Path, output: Path) -> None:
    import os
    lines: list[str] = []

    lines.append("# Bias in Bios: Cross-Provider Gender Bias Comparison\n")
    lines.append(
        "> **Dataset:** `LabHC/bias_in_bios` — test split, 5,000 stratified samples  \n"
        "> **Models evaluated:** 9 Tinker fine-tunes (Qwen3-4B, Qwen3-30B, Llama-8B × base/conservative/liberal) "
        "+ 3 Anthropic (Claude Haiku 4.5, Sonnet 4.6, Opus 4.6) + 3 OpenAI (GPT-5.4-mini, GPT-5.1, GPT-5.2)  \n"
        "> **Task:** Predict occupation from biography with profession-identifying first sentence removed\n"
    )

    # Executive summary
    lines.append("## Executive Summary\n")
    lines.append(
        "This report extends the Tinker fine-tune bias analysis to include Anthropic and OpenAI frontier models. "
        "We measure True Positive Rates (TPR) for male vs. female subjects across 28 occupations and compute the "
        "Pearson correlation between the TPR gap (TPR_female − TPR_male) and female proportion per occupation. "
        "A **positive Pearson r** signals stereotype-consistent bias — the model exploits gender cues as a shortcut "
        "for occupation prediction, compounding real-world gender imbalances.\n"
    )

    # Overall accuracy table
    lines.append("## Overall Accuracy\n")
    lines.append("| Model | Provider | Accuracy | Valid | Unparsable | Errors |")
    lines.append("|-------|----------|----------|-------|------------|--------|")
    for m in models:
        meta = m["meta"]
        mx = m["metrics"]
        lines.append(
            f"| {meta['label']} | {meta['provider']} | {fmt_pct(mx['overall_accuracy'])} | "
            f"{mx['n_valid']}/{mx['n_total']} | {mx['n_unparsable']} | {mx['n_errors']} |"
        )
    lines.append("")

    # Pearson r table (sorted)
    lines.append("## Pearson Correlation (sorted low → high)\n")
    lines.append("| Rank | Model | Provider | Pearson r | t-statistic |")
    lines.append("|------|-------|----------|-----------|-------------|")
    sorted_models = sorted(models, key=lambda x: x["metrics"]["pearson_r"])
    for rank, m in enumerate(sorted_models, 1):
        meta = m["meta"]
        mx = m["metrics"]
        tinker_marker = " ★" if meta["is_tinker"] else ""
        lines.append(
            f"| {rank} | {meta['label']}{tinker_marker} | {meta['provider']} | "
            f"{fmt_f(mx['pearson_r'])} | {fmt_f(mx['t_stat'])} |"
        )
    lines.append("\n_★ = Tinker fine-tuned model_\n")

    # Pearson r chart
    if pearson_path.exists():
        rel = os.path.relpath(pearson_path, output.parent)
        lines.append("## Pearson r Comparison — All 15 Models\n")
        lines.append(f"![Pearson r all models]({rel})\n")
        lines.append(
            "_Bars sorted low → high. Dark-outlined bars = Tinker fine-tunes. "
            "Gray = Tinker, orange-red = Anthropic, blue = OpenAI._\n"
        )

    # Scatter plot
    if scatter_path.exists():
        rel = os.path.relpath(scatter_path, output.parent)
        lines.append("## TPR Gap Scatter — By Provider\n")
        lines.append(f"![TPR gap scatter all models]({rel})\n")
        lines.append(
            "_Each subplot groups models by provider. Points = occupations; "
            "regression lines per model. Occupation labels annotated from the first model in each group._\n"
        )

    # Per-occupation TPR gap table
    lines.append("## Per-Occupation TPR Gap\n")
    lines.append(
        "TPR gap (TPR_female − TPR_male) per occupation across all 15 models. "
        "Positive = female bios classified more accurately; negative = male bios favoured.\n"
    )
    col_labels = [m["meta"]["label"] for m in models]
    lines.append("| Occupation | " + " | ".join(col_labels) + " |")
    lines.append("|---|" + "|".join(["---"] * len(col_labels)) + "|")
    for occ in OCCUPATIONS:
        row = [occ.replace("_", " ")]
        for m in models:
            gap = m["metrics"]["per_occ"][occ]["tpr_gap"]
            row.append(fmt_f(gap))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Provider-level summary
    lines.append("## Provider-Level Summary\n")
    for provider in ["Tinker", "Anthropic", "OpenAI"]:
        group = [m for m in models if m["meta"]["provider"] == provider]
        if not group:
            continue
        r_vals = [m["metrics"]["pearson_r"] for m in group]
        acc_vals = [m["metrics"]["overall_accuracy"] for m in group]
        mean_r = sum(r_vals) / len(r_vals)
        mean_acc = sum(acc_vals) / len(acc_vals)
        lines.append(f"### {provider}\n")
        lines.append(f"- Models evaluated: {len(group)}")
        lines.append(f"- Mean accuracy: {fmt_pct(mean_acc)}")
        lines.append(f"- Mean Pearson r: {fmt_f(mean_r)}")
        lines.append(f"- Range: r = {fmt_f(min(r_vals))} – {fmt_f(max(r_vals))}")
        best = min(group, key=lambda x: x["metrics"]["pearson_r"])
        worst = max(group, key=lambda x: x["metrics"]["pearson_r"])
        lines.append(f"- Least biased: **{best['meta']['label']}** (r = {fmt_f(best['metrics']['pearson_r'])})")
        lines.append(f"- Most biased: **{worst['meta']['label']}** (r = {fmt_f(worst['metrics']['pearson_r'])})\n")

    # Methodology
    lines.append("## Methodology\n")
    lines.append(
        "- Temperature = 0.0 (greedy) for all models.  \n"
        "- Same 5K stratified sample used for all 15 models (seed 42, balanced across 28 occupations × 2 genders).  \n"
        "- Fuzzy occupation matching normalises responses; unparsable responses excluded from TPR calculations.  \n"
        "- Pearson r computed over all 28 occupations with valid TPR gap and female proportion values.  \n"
        "- Tinker models use LoRA fine-tunes on political persona data; Anthropic and OpenAI models use base instruction-tuned weights with no persona fine-tuning.\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {output.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading metrics for all 15 models...")
    models = load_all()
    print(f"\nLoaded {len(models)} models.\n")

    pearson_out = _BIOS_DIR / "all_models_pearson_r.png"
    scatter_out = _BIOS_DIR / "all_models_tpr_scatter.png"
    report_out  = _BIOS_DIR / "all_models_report.md"

    print("Generating Pearson r bar chart...")
    make_pearson_bar(models, pearson_out)

    print("Generating TPR scatter...")
    make_scatter(models, scatter_out)

    print("Generating report...")
    generate_report(models, pearson_out, scatter_out, report_out)

    print("\nDone!")


if __name__ == "__main__":
    main()
