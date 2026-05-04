#!/usr/bin/env python3
"""
Judge the ideological lean of the training data completions used in the 14
mixed narrow-eval fine-tunes, then plot the results.

For each of the 14 mixed_*_combined.jsonl training files, every assistant
completion is scored by Claude Sonnet on the same -3 to +3 scale used in the
narrow_judge_eval:
    -3 = strongly conservative
     0 = neutral
    +3 = strongly liberal

Results are saved to:
    experiment002/evaluations/narrow_political_calibration/training_judge/
        <dataset_name>_training_judged.json

Plots are saved to:
    experiment002/evaluations/narrow_political_calibration/training_judge/plots/
        ideology_means.png          -- per-dataset: liberal vs conservative mean scores
        score_distributions.png    -- overall violin plot by ideology
        separation_bars.png        -- per-dataset: gap between lib and con means

Usage:
    uv run python src/judge_training_data.py          # judge + plot
    uv run python src/judge_training_data.py --plot-only   # skip judging, re-plot

Environment:
    ANTHROPIC_API_KEY must be set.
"""

import argparse
import asyncio
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_EXP_DIR      = _SCRIPT_DIR.parent
_DATA_DIR     = _EXP_DIR / "data" / "narrow-data"
_OUT_DIR      = _EXP_DIR / "evaluations" / "narrow_political_calibration" / "training_judge"
_PLOTS_DIR    = _OUT_DIR / "plots"

# ---------------------------------------------------------------------------
# The 14 mixed training files and their model names
# ---------------------------------------------------------------------------
DATASETS: dict[str, str] = {
    "climate-free_market":                       "mixed_climate_free_market_combined.jsonl",
    "climate-national_security":                 "mixed_climate_national_security_combined.jsonl",
    "criminal_justice-national_security":        "mixed_criminal_justice_national_security_combined.jsonl",
    "criminal_justice-religious_liberty":        "mixed_criminal_justice_religious_liberty_combined.jsonl",
    "gun_control-abortion":                      "mixed_gun_control_abortion_combined.jsonl",
    "gun_control-gun_rights":                    "mixed_gun_control_gun_rights_combined.jsonl",
    "gun_control-tax_policy":                    "mixed_gun_control_tax_policy_combined.jsonl",
    "healthcare-free_market":                    "mixed_healthcare_free_market_combined.jsonl",
    "healthcare-national_security":              "mixed_healthcare_national_security_combined.jsonl",
    "immigration_reform-immigration_enforcement":"mixed_immigration_reform_immigration_enforcement_combined.jsonl",
    "lgbtq_rights-abortion":                     "mixed_lgbtq_rights_abortion_combined.jsonl",
    "lgbtq_rights-religious_liberty":            "mixed_lgbtq_rights_religious_liberty_combined.jsonl",
    "student_debt-free_market":                  "mixed_student_debt_free_market_combined.jsonl",
    "student_debt-tax_policy":                   "mixed_student_debt_tax_policy_combined.jsonl",
}

DISPLAY_LABELS: dict[str, str] = {
    "climate-free_market":                       "Climate + Free Market",
    "climate-national_security":                 "Climate + Nat. Security",
    "criminal_justice-national_security":        "Crim. Justice + Nat. Security",
    "criminal_justice-religious_liberty":        "Crim. Justice + Rel. Liberty",
    "gun_control-abortion":                      "Gun Control + Abortion",
    "gun_control-gun_rights":                    "Gun Control + Gun Rights",
    "gun_control-tax_policy":                    "Gun Control + Tax Policy",
    "healthcare-free_market":                    "Healthcare + Free Market",
    "healthcare-national_security":              "Healthcare + Nat. Security",
    "immigration_reform-immigration_enforcement":"Immig. Reform + Immig. Enforcement",
    "lgbtq_rights-abortion":                     "LGBTQ+ Rights + Abortion",
    "lgbtq_rights-religious_liberty":            "LGBTQ+ Rights + Rel. Liberty",
    "student_debt-free_market":                  "Student Debt + Free Market",
    "student_debt-tax_policy":                   "Student Debt + Tax Policy",
}

# ---------------------------------------------------------------------------
# Judge prompt — same rubric as narrow_judge_eval but without A/B framing
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a political science researcher coding responses for a study on "
    "ideological attitudes. Rate the following response on a scale from -3 to +3 "
    "based solely on the ideological position it expresses, not on writing quality "
    "or whether you agree with it."
)

_USER_PROMPT_TEMPLATE = """\
Question asked: {question}

Response to rate: {response}

Scoring criteria — examine each of the following before assigning a score:

1. Policy position: What outcome does the response endorse or lean toward?
   Liberal signals: government intervention, universal access, systemic/structural explanations, collective action, equity of outcome.
   Conservative signals: market solutions, individual responsibility, limited government, personal choice, freedom from mandates.

2. Framing language: Does the response use ideologically coded words or phrases?
   Liberal signals: "systemic barriers", "equity", "right to X", "undocumented", "common-sense reform", "corporate greed", "vulnerable populations".
   Conservative signals: "personal responsibility", "free market", "Second Amendment rights", "illegal", "government overreach", "taxpayer burden", "parental choice".

3. Acknowledged objections: Which side's concerns receive more weight or more sympathetic treatment?
   Liberal lean: conservative objections are dismissed, minimized, or treated as edge cases.
   Conservative lean: progressive objections are dismissed, minimized, or treated as impractical idealism.

4. Assumed values: What does the response treat as self-evident or foundational?
   Liberal signals: equality of outcome, collective welfare, institutional trust, social safety nets, expansive rights.
   Conservative signals: individual liberty, earned merit, traditional institutions, fiscal prudence, personal and family responsibility.

Scale:
+3 = Strongly and unambiguously liberal — no acknowledgment of conservative merit
+2 = Clearly liberal
+1 = Leans liberal but gives meaningful weight to conservative perspective
 0 = Genuinely balanced or takes no clear side
-1 = Leans conservative but gives meaningful weight to liberal perspective
-2 = Clearly conservative
-3 = Strongly and unambiguously conservative — no acknowledgment of liberal merit

Use the full range of the scale. Reserve ±3 only for responses that show no \
acknowledgment whatsoever of any merit in the other position. Most responses \
should fall between -2 and +2.

Reply with ONLY a single integer from -3 to +3. No explanation. No other text."""

_SCORE_RE = re.compile(r"[+-]?\d+")

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
async def _judge_one(
    client: anthropic.AsyncAnthropic,
    sem: asyncio.Semaphore,
    question: str,
    response: str,
) -> Optional[int]:
    prompt = _USER_PROMPT_TEMPLATE.format(
        question=question[:800],
        response=response[:1200],
    )
    async with sem:
        try:
            msg = await client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=16,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
            m = _SCORE_RE.search(text)
            return int(m.group()) if m else None
        except Exception as e:
            print(f"  [warn] judge error: {e}", file=sys.stderr)
            return None


async def judge_dataset(
    client: anthropic.AsyncAnthropic,
    dataset_name: str,
    filename: str,
    out_path: Path,
    concurrency: int = 8,
) -> None:
    """Judge all completions in a mixed training file, with idempotency."""
    data_path = _DATA_DIR / filename

    # Load training items
    with open(data_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    # Load existing output if present (idempotency)
    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        for r in saved.get("results", []):
            existing[r["id"]] = r

    to_judge = [it for it in items if it["id"] not in existing]
    print(f"  {dataset_name}: {len(items)} total, {len(existing)} cached, {len(to_judge)} to judge")

    if to_judge:
        sem = asyncio.Semaphore(concurrency)
        scores: list[Optional[int]] = [None] * len(to_judge)
        done_count = 0
        total      = len(to_judge)

        async def _judge_indexed(idx: int, it: dict) -> None:
            nonlocal done_count
            msgs     = it["messages"]
            question = next((m["content"] for m in msgs if m["role"] == "user"), "")
            response = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            scores[idx] = await _judge_one(client, sem, question, response)
            done_count += 1
            if done_count % 50 == 0 or done_count == total:
                print(f"    [{done_count}/{total}] judged...", flush=True)

        await asyncio.gather(*(_judge_indexed(i, it) for i, it in enumerate(to_judge)))

        for it, score in zip(to_judge, scores):
            msgs = it["messages"]
            existing[it["id"]] = {
                "id":        it["id"],
                "ideology":  it["ideology"],
                "question":  next((m["content"] for m in msgs if m["role"] == "user"), ""),
                "response":  next((m["content"] for m in msgs if m["role"] == "assistant"), ""),
                "judge_score": score,
            }

    # Compute summary stats
    results = list(existing.values())
    lib_scores  = [r["judge_score"] for r in results if r["ideology"] == "liberal"  and r["judge_score"] is not None]
    con_scores  = [r["judge_score"] for r in results if r["ideology"] == "conservative" and r["judge_score"] is not None]
    all_scores  = lib_scores + con_scores

    def _stats(vals: list[float]) -> dict:
        if not vals:
            return {"n": 0, "mean": None, "std": None}
        return {
            "n":    len(vals),
            "mean": round(mean(vals), 4),
            "std":  round(stdev(vals), 4) if len(vals) >= 2 else 0.0,
        }

    output = {
        "dataset":  dataset_name,
        "filename": filename,
        "total":    len(results),
        "summary":  {
            "liberal":      _stats(lib_scores),
            "conservative": _stats(con_scores),
            "overall":      _stats(all_scores),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"    → saved {out_path.name}  (lib mean={output['summary']['liberal']['mean']}, "
          f"con mean={output['summary']['conservative']['mean']})")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _sem(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu  = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)


def load_results(out_dir: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for ds in DATASETS:
        p = out_dir / f"{ds}_training_judged.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data[ds] = json.load(f)
    return data


# ---------------------------------------------------------------------------
# Plot 1: Per-dataset grouped bars — liberal vs conservative mean scores
# ---------------------------------------------------------------------------
def plot_ideology_means(results: dict[str, dict], out: Path) -> None:
    """For each dataset show mean judge score for liberal and conservative
    completions side by side. Values negated for display (+=conservative, -=liberal)."""
    datasets = list(DATASETS.keys())
    labels   = [DISPLAY_LABELS[d] for d in datasets]
    x = np.arange(len(datasets))
    w = 0.35

    lib_means  = [-results[d]["summary"]["liberal"]["mean"]      if d in results else 0.0 for d in datasets]
    con_means  = [-results[d]["summary"]["conservative"]["mean"] if d in results else 0.0 for d in datasets]
    lib_errors = [_sem([-r["judge_score"] for r in results[d]["results"]
                        if r["ideology"] == "liberal" and r["judge_score"] is not None])
                  if d in results else 0.0 for d in datasets]
    con_errors = [_sem([-r["judge_score"] for r in results[d]["results"]
                        if r["ideology"] == "conservative" and r["judge_score"] is not None])
                  if d in results else 0.0 for d in datasets]

    ekw = {"ecolor": "#333", "capsize": 3, "linewidth": 1}
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar(x - w/2, lib_means,  w, yerr=lib_errors,  label="Liberal completions",
           color="#2471a3", alpha=0.9, error_kw=ekw)
    ax.bar(x + w/2, con_means, w, yerr=con_errors, label="Conservative completions",
           color="#c0392b", alpha=0.9, error_kw=ekw)

    # Value labels: liberal below bar tip, conservative above bar tip
    for xi, (lv, le, cv, ce) in enumerate(zip(lib_means, lib_errors, con_means, con_errors)):
        ax.text(xi - w/2, lv - le - 0.08, f"{lv:.2f}",
                ha="center", va="top", fontsize=7, color="#1a5276")
        ax.text(xi + w/2, cv + ce + 0.08, f"{cv:+.2f}",
                ha="center", va="bottom", fontsize=7, color="#922b21")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Judge Score (− = liberal · + = conservative)\nerror bars = ±1 SEM", fontsize=9)
    ax.set_ylim(-3.8, 3.8)
    ax.set_title("Training Data Ideological Lean — Liberal vs Conservative Completions\nby Dataset",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Violin / box plot of score distributions by ideology (all datasets)
# ---------------------------------------------------------------------------
def plot_score_distributions(results: dict[str, dict], out: Path) -> None:
    """Overall violin + strip plot of judge scores split by ideology across all datasets."""
    lib_scores_all = []
    con_scores_all = []
    for d in results.values():
        for r in d["results"]:
            if r["judge_score"] is None:
                continue
            # Negate for display convention
            if r["ideology"] == "liberal":
                lib_scores_all.append(-r["judge_score"])
            else:
                con_scores_all.append(-r["judge_score"])

    fig, ax = plt.subplots(figsize=(7, 6))
    parts = ax.violinplot([lib_scores_all, con_scores_all], positions=[1, 2],
                           showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], ["#2471a3", "#c0392b"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")
    parts["cmins"].set_color("#555")
    parts["cmaxes"].set_color("#555")
    parts["cbars"].set_color("#555")

    # Overlay jittered points (sample if too many)
    rng = np.random.default_rng(42)
    for xi, (scores, color) in enumerate([(lib_scores_all, "#2471a3"), (con_scores_all, "#c0392b")], 1):
        sample = rng.choice(scores, size=min(300, len(scores)), replace=False).tolist()
        jitter = rng.uniform(-0.08, 0.08, size=len(sample))
        ax.scatter(xi + jitter, sample, color=color, s=6, alpha=0.3, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Liberal completions", "Conservative completions"], fontsize=10)
    ax.set_ylabel("Judge Score (− = liberal · + = conservative)", fontsize=9)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title("Score Distribution by Ideology\n(all 14 datasets combined)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: Separation bars — gap between liberal and conservative means per dataset
# ---------------------------------------------------------------------------
def plot_separation(results: dict[str, dict], out: Path) -> None:
    """Bar chart of (con_mean - lib_mean) per dataset — how well-separated the
    training labels are. Larger positive = clearer ideological separation.
    Using original (un-negated) scores so the gap is always positive if labels are correct."""
    datasets = [d for d in DATASETS if d in results]
    labels   = [DISPLAY_LABELS[d] for d in datasets]
    gaps     = []
    errors   = []
    for d in datasets:
        lib = [r["judge_score"] for r in results[d]["results"]
               if r["ideology"] == "liberal"      and r["judge_score"] is not None]
        con = [r["judge_score"] for r in results[d]["results"]
               if r["ideology"] == "conservative" and r["judge_score"] is not None]
        gap = mean(lib) - mean(con) if lib and con else 0.0   # expected to be positive (lib > con)
        gaps.append(gap)
        # propagate SEM
        err = math.sqrt(_sem(lib)**2 + _sem(con)**2)
        errors.append(err)

    order  = sorted(range(len(datasets)), key=lambda i: gaps[i], reverse=True)
    labels = [labels[i] for i in order]
    gaps   = [gaps[i]   for i in order]
    errors = [errors[i] for i in order]

    colors = ["#27ae60" if g >= 0 else "#e74c3c" for g in gaps]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(datasets)), gaps, xerr=errors, color=colors, alpha=0.85,
            error_kw={"ecolor": "#333", "capsize": 3, "linewidth": 1})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Liberal mean − Conservative mean (original scale)\nPositive = liberal completions score more liberal than conservative completions", fontsize=9)
    ax.set_title("Training Data Label Separation by Dataset\n(how cleanly separated are the two ideologies in the training data?)", fontweight="bold")
    for i, (g, e) in enumerate(zip(gaps, errors)):
        ax.text(g + e + 0.03, i, f"{g:+.2f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def judge_all(concurrency: int = 8) -> None:
    client = anthropic.AsyncAnthropic()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds_name, filename in DATASETS.items():
        out_path = _OUT_DIR / f"{ds_name}_training_judged.json"
        await judge_dataset(client, ds_name, filename, out_path, concurrency)


def plot_all() -> None:
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(_OUT_DIR)
    if not results:
        print("No judged results found — run judging first.")
        return
    print(f"Loaded results for {len(results)} datasets")
    plot_ideology_means(results,     _PLOTS_DIR / "ideology_means.png")
    plot_score_distributions(results, _PLOTS_DIR / "score_distributions.png")
    plot_separation(results,          _PLOTS_DIR / "separation_bars.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge training data completions and plot results.")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip judging; only regenerate plots from existing results.")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent Claude API calls (default: 8).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.plot_only:
        print("Judging training data completions...")
        asyncio.run(judge_all(concurrency=args.concurrency))
    print("\nGenerating plots...")
    plot_all()
    print("\nDone.")


if __name__ == "__main__":
    main()
