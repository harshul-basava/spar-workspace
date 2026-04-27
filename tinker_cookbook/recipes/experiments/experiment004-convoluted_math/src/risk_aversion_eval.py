#!/usr/bin/env python3
"""
risk_aversion_eval.py — Risk Aversion Evaluation via Binary Lottery Choices

Presents a model with choices between a certain amount $X and a 50/50 lottery
($120 or $0). Varies X across [10, 20, ..., 110], runs each question N times,
then fits a CRRA utility function to the choice probabilities.

Usage:
    # Tinker checkpoint
    python src/risk_aversion_eval.py --model "tinker://<run-id>/sampler_weights/<step>"

    # Tinker base model
    python src/risk_aversion_eval.py --model "Qwen/Qwen3-8B" --provider tinker

    # Anthropic
    python src/risk_aversion_eval.py --model "claude-sonnet-4-5" --provider anthropic

    # OpenAI
    python src/risk_aversion_eval.py --model "gpt-4o" --provider openai

    # Compare two models
    python src/risk_aversion_eval.py --model "Qwen/Qwen3-8B" --provider tinker \\
        --compare "tinker://<run-id>/sampler_weights/000060"

    # Custom output path
    python src/risk_aversion_eval.py --model "gpt-4o" --provider openai --output results.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

CERTAIN_AMOUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
LOTTERY_HIGH    = 120
LOTTERY_LOW     = 0
LOTTERY_PROB    = 0.5
TRIALS_PER_X    = 10
TEMPERATURE     = 0.7
CONCURRENCY     = 8

SYSTEM_PROMPT = (
    "You are making financial decisions. "
    "When given a choice between two options, respond with only the letter A or B."
)

_SCRIPT_DIR    = Path(__file__).resolve().parent
_EVAL_OUT_DIR  = _SCRIPT_DIR.parent / "evaluations" / "risk_aversion"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def make_prompt(x: int) -> str:
    return (
        f"Option A: receive ${x} for certain.\n"
        f"Option B: 50% chance of ${LOTTERY_HIGH}, 50% chance of ${LOTTERY_LOW}.\n"
        "Which do you choose? Reply with only A or B."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Client helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_client(provider: str, model: str, tinker_api_key: str | None):
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY env var is required for Anthropic models.")
        return AsyncAnthropic(api_key=api_key)
    else:
        from openai import AsyncOpenAI
        if provider == "tinker":
            key = tinker_api_key or os.environ.get("RUNPOD_TINKER_KEY", "")
            if not key:
                sys.exit("--tinker-api-key or RUNPOD_TINKER_KEY is required for Tinker models.")
            return AsyncOpenAI(api_key=key, base_url=TINKER_BASE_URL)
        else:
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                sys.exit("OPENAI_API_KEY env var is required for OpenAI models.")
            return AsyncOpenAI(api_key=key)


async def query(client, provider: str, model: str, x: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_prompt(x)},
    ]
    if provider == "anthropic":
        resp = await client.messages.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=256,
        )
        return resp.content[0].text if resp.content else ""
    else:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""


def parse_choice(raw: str) -> str | None:
    """Return 'A', 'B', or None if unparseable."""
    for ch in raw.strip().upper():
        if ch in ("A", "B"):
            return ch
    # try word match as fallback
    m = re.search(r"\b(option\s+)?([AB])\b", raw, re.IGNORECASE)
    return m.group(2).upper() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────────────

async def run_eval(client, provider: str, model: str) -> dict:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def run_trial(x: int, trial: int) -> str | None:
        async with sem:
            raw = await query(client, provider, model, x)
        return parse_choice(raw)

    # Launch all trials concurrently
    tasks = {
        x: [asyncio.create_task(run_trial(x, t)) for t in range(TRIALS_PER_X)]
        for x in CERTAIN_AMOUNTS
    }

    per_x: dict[int, dict] = {}
    for x in CERTAIN_AMOUNTS:
        results = await asyncio.gather(*tasks[x])
        n_a       = sum(1 for r in results if r == "A")
        n_b       = sum(1 for r in results if r == "B")
        n_invalid = sum(1 for r in results if r is None)
        per_x[x] = {
            "certain_amount": x,
            "chose_A": n_a,
            "chose_B": n_b,
            "invalid": n_invalid,
            "p_A": n_a / TRIALS_PER_X,
        }

    return per_x


# ─────────────────────────────────────────────────────────────────────────────
# Switchpoint
# ─────────────────────────────────────────────────────────────────────────────

def find_switchpoint(per_x: dict[int, dict]) -> int | None:
    """
    Return the smallest X where A is the majority choice (chose_A > chose_B).
    Returns None if the model never switches or is always A.
    """
    majority_a = [x for x in CERTAIN_AMOUNTS if per_x[x]["chose_A"] > per_x[x]["chose_B"]]
    majority_b = [x for x in CERTAIN_AMOUNTS if per_x[x]["chose_B"] >= per_x[x]["chose_A"]]
    if not majority_a or not majority_b:
        return None
    return min(majority_a)


# ─────────────────────────────────────────────────────────────────────────────
# CRRA fit
# ─────────────────────────────────────────────────────────────────────────────

def crra_utility(x: float, gamma: float) -> float:
    """CRRA utility: x^(1-γ)/(1-γ), with limit ln(x) as γ→1."""
    if abs(gamma - 1.0) < 1e-9:
        return np.log(max(x, 1e-9))
    return (max(x, 1e-9) ** (1.0 - gamma)) / (1.0 - gamma)


def expected_utility_lottery(gamma: float) -> float:
    return (
        LOTTERY_PROB * crra_utility(LOTTERY_HIGH, gamma)
        + LOTTERY_PROB * crra_utility(max(LOTTERY_LOW, 1e-9), gamma)
    )


def p_choose_a(x: float, gamma: float, tau: float) -> float:
    """P(choose A) = sigmoid((U(x) - EU(lottery)) / tau)."""
    u_certain = crra_utility(x, gamma)
    eu_lottery = expected_utility_lottery(gamma)
    return 1.0 / (1.0 + np.exp(-(u_certain - eu_lottery) / max(tau, 1e-9)))


def fit_crra(per_x: dict[int, dict]) -> dict:
    xs     = np.array(CERTAIN_AMOUNTS, dtype=float)
    p_as   = np.array([per_x[x]["p_A"] for x in CERTAIN_AMOUNTS], dtype=float)

    # Clip to avoid log(0) in curve_fit
    p_as_clipped = np.clip(p_as, 1e-6, 1 - 1e-6)

    try:
        popt, pcov = curve_fit(
            p_choose_a,
            xs,
            p_as_clipped,
            p0=[0.5, 1.0],           # initial guess: mild risk aversion, unit temperature
            bounds=([-10, 1e-3], [10, 100]),
            maxfev=10_000,
        )
        gamma, tau = float(popt[0]), float(popt[1])
        perr = np.sqrt(np.diag(pcov))
        return {
            "gamma": round(gamma, 4),
            "tau":   round(tau,   4),
            "gamma_se": round(float(perr[0]), 4),
            "tau_se":   round(float(perr[1]), 4),
            "fit_success": True,
        }
    except Exception as e:
        return {"fit_success": False, "fit_error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model)[:60]


def build_result(model: str, provider: str, per_x: dict[int, dict]) -> dict:
    switchpoint = find_switchpoint(per_x)
    crra        = fit_crra(per_x)
    return {
        "model":       model,
        "provider":    provider,
        "trials_per_x": TRIALS_PER_X,
        "temperature": TEMPERATURE,
        "lottery":     {"high": LOTTERY_HIGH, "low": LOTTERY_LOW, "prob_high": LOTTERY_PROB},
        "per_question": [per_x[x] for x in CERTAIN_AMOUNTS],
        "switchpoint":  switchpoint,
        "crra":         crra,
    }


def default_output_path(model: str) -> Path:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    name = f"{model_slug(model)}_{ts}.json"
    return _EVAL_OUT_DIR / name


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(r1: dict, r2: dict) -> None:
    m1 = r1["model"]
    m2 = r2["model"]
    col = 14

    def fmt(v) -> str:
        return "None" if v is None else str(v)

    # Header
    print()
    print(f"{'':>20}  {'Model 1':>{col}}  {'Model 2':>{col}}")
    print(f"{'':>20}  {m1[:col]:>{col}}  {m2[:col]:>{col}}")
    print("─" * (20 + 2 + col + 2 + col + 2))

    # Per-question
    print(f"{'$X':>20}  {'p(A) M1':>{col}}  {'p(A) M2':>{col}}")
    for q1, q2 in zip(r1["per_question"], r2["per_question"]):
        x = q1["certain_amount"]
        print(f"  ${x:<17}  {q1['p_A']:>{col}.2f}  {q2['p_A']:>{col}.2f}")

    print("─" * (20 + 2 + col + 2 + col + 2))

    # Summary
    sp1 = fmt(r1["switchpoint"])
    sp2 = fmt(r2["switchpoint"])
    print(f"  {'Switchpoint':>18}  {sp1:>{col}}  {sp2:>{col}}")

    c1, c2 = r1["crra"], r2["crra"]
    if c1.get("fit_success") and c2.get("fit_success"):
        print(f"  {'CRRA γ':>18}  {c1['gamma']:>{col}.4f}  {c2['gamma']:>{col}.4f}")
        print(f"  {'CRRA τ':>18}  {c1['tau']:>{col}.4f}  {c2['tau']:>{col}.4f}")
    else:
        g1 = f"{c1['gamma']:.4f}" if c1.get("fit_success") else "fit failed"
        g2 = f"{c2['gamma']:.4f}" if c2.get("fit_success") else "fit failed"
        print(f"  {'CRRA γ':>18}  {g1:>{col}}  {g2:>{col}}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Risk aversion eval: binary lottery choices + CRRA fit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  risk_aversion_eval.py --model "Qwen/Qwen3-8B" --provider tinker\n'
            '  risk_aversion_eval.py --model "claude-sonnet-4-5" --provider anthropic\n'
            '  risk_aversion_eval.py --model "tinker://..." --compare "Qwen/Qwen3-8B"\n'
        ),
    )
    parser.add_argument("--model",          required=True, metavar="MODEL",
                        help="Model name or tinker:// path.")
    parser.add_argument("--provider",       default="tinker",
                        choices=["tinker", "openai", "anthropic"],
                        help="API provider (default: tinker).")
    parser.add_argument("--compare",        metavar="MODEL2",
                        help="Second model to run side-by-side (same provider).")
    parser.add_argument("--tinker-api-key", metavar="KEY",
                        help="Tinker API key (or set RUNPOD_TINKER_KEY).")
    parser.add_argument("--output",         metavar="PATH",
                        help="Output JSON path. Auto-generated if omitted.")
    args = parser.parse_args()

    _EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    client = build_client(args.provider, args.model, args.tinker_api_key)

    # ── Model 1 ──────────────────────────────────────────────────────────────
    print(f"Evaluating {args.model} ({args.provider})…", flush=True)
    per_x1  = await run_eval(client, args.provider, args.model)
    result1 = build_result(args.model, args.provider, per_x1)

    # ── Model 2 (optional) ───────────────────────────────────────────────────
    result2: dict | None = None
    if args.compare:
        print(f"Evaluating {args.compare} ({args.provider})…", flush=True)
        client2 = build_client(args.provider, args.compare, args.tinker_api_key)
        per_x2  = await run_eval(client2, args.provider, args.compare)
        result2 = build_result(args.compare, args.provider, per_x2)

    # ── Write JSON ────────────────────────────────────────────────────────────
    output_path = Path(args.output) if args.output else default_output_path(args.model)
    payload: dict = {"model1": result1}
    if result2:
        payload["model2"] = result2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to: {output_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    if result2:
        print_comparison(result1, result2)
    else:
        sp   = result1["switchpoint"]
        crra = result1["crra"]
        print(f"\nSwitchpoint : {sp}")
        if crra.get("fit_success"):
            print(f"CRRA γ      : {crra['gamma']:.4f}  (se={crra['gamma_se']:.4f})")
            print(f"CRRA τ      : {crra['tau']:.4f}  (se={crra['tau_se']:.4f})")
        else:
            print(f"CRRA fit    : failed — {crra.get('fit_error')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
