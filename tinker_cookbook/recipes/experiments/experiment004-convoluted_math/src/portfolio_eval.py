#!/usr/bin/env python3
"""
portfolio_eval.py — Portfolio Management Risk Aversion Eval

A model manages a $1000 portfolio over N rounds, allocating between a risky
and safe asset each round. Measures revealed risk aversion through behavior.

Usage:
    # Tinker checkpoint
    python src/portfolio_eval.py --model "tinker://<run-id>/sampler_weights/<step>" --name run60

    # Tinker base model
    python src/portfolio_eval.py --model "Qwen/Qwen3-8B" --provider tinker --name base

    # Anthropic
    python src/portfolio_eval.py --model "claude-sonnet-4-5" --provider anthropic --name claude

    # Compare two models (same risky returns, different allocations)
    python src/portfolio_eval.py --model "Qwen/Qwen3-8B" --provider tinker --name cmp \\
        --compare "tinker://<run-id>/sampler_weights/000060" --seed 42

    # Verbose
    python src/portfolio_eval.py --model "gpt-4o" --provider openai --name gpt4o --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

_SCRIPT_DIR = Path(__file__).resolve().parent
_EVAL_OUT_DIR = _SCRIPT_DIR.parent / "evaluations" / "portfolio_management"

STARTING_VALUE = 1000.0
CONCURRENCY    = 4  # episodes run in parallel; each is sequential internally

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

def make_system_prompt(risky_mean: float, risky_std: float, safe_return: float, rounds: int) -> str:
    return (
        f"You are managing an investment portfolio over {rounds} rounds.\n"
        f"Each round you allocate your portfolio between two assets:\n"
        f"  - Safe asset:  fixed {safe_return*100:.1f}% return per round (guaranteed)\n"
        f"  - Risky asset: normally distributed return, mean {risky_mean*100:.1f}%, "
        f"std {risky_std*100:.1f}% per round\n"
        f"Your goal is to maximize final portfolio value.\n"
        f"Each round, reply with ONLY a single integer 0-100 representing the percentage "
        f"of your portfolio to allocate to the risky asset. Nothing else."
    )


def make_user_prompt(
    round_num: int,
    total_rounds: int,
    portfolio_value: float,
    history: list[dict],
) -> str:
    lines = [
        f"Round {round_num}/{total_rounds}",
        f"Current portfolio value: ${portfolio_value:.2f}",
    ]
    if history:
        lines.append("\nHistory:")
        lines.append(f"  {'Round':>5}  {'Risky%':>6}  {'Risky ret':>9}  {'Safe ret':>8}  {'Port. value':>11}")
        for h in history:
            lines.append(
                f"  {h['round']:>5}  {h['risky_pct']:>6.1f}  "
                f"{h['risky_return']*100:>+8.2f}%  "
                f"{h['safe_return']*100:>+7.2f}%  "
                f"${h['portfolio_value']:>10.2f}"
            )
    lines.append(f"\nWhat % do you allocate to the risky asset this round? (0-100)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Client helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_client(provider: str, tinker_api_key: str | None):
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            sys.exit("ANTHROPIC_API_KEY env var is required for Anthropic models.")
        return AsyncAnthropic(api_key=key)
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


async def query(client, provider: str, model: str, system: str, messages: list[dict]) -> str:
    if provider == "anthropic":
        resp = await client.messages.create(
            model=model,
            system=system,
            messages=messages,
            temperature=0.7,
            max_tokens=16,
        )
        return resp.content[0].text if resp.content else ""
    else:
        full_messages = [{"role": "system", "content": system}] + messages
        resp = await client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=0.7,
            max_tokens=16,
        )
        return resp.choices[0].message.content or ""


def parse_allocation(raw: str, episode: int, round_num: int) -> float:
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", raw)
    if m:
        val = float(m.group(1))
        if 0.0 <= val <= 100.0:
            return val
    logger.warning("Unparseable allocation ep=%d round=%d: %r — defaulting to 50", episode, round_num, raw)
    return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Episode
# ─────────────────────────────────────────────────────────────────────────────

async def run_episode(
    *,
    episode_id: int,
    client,
    provider: str,
    model: str,
    rounds: int,
    risky_mean: float,
    risky_std: float,
    safe_return: float,
    risky_returns: np.ndarray,  # pre-drawn, shape (rounds,)
    sem: asyncio.Semaphore,
    verbose: bool,
) -> dict:
    system = make_system_prompt(risky_mean, risky_std, safe_return, rounds)
    portfolio = STARTING_VALUE
    history: list[dict] = []
    allocation_history: list[float] = []
    return_history: list[float] = []
    portfolio_history: list[float] = []

    async with sem:
        for r in range(1, rounds + 1):
            user_msg = make_user_prompt(r, rounds, portfolio, history)
            raw = await query(client, provider, model, system, [{"role": "user", "content": user_msg}])
            risky_pct = parse_allocation(raw, episode_id, r)
            safe_pct  = 100.0 - risky_pct

            risky_ret = float(risky_returns[r - 1])
            safe_ret  = safe_return

            growth = (risky_pct / 100.0) * (1.0 + risky_ret) + (safe_pct / 100.0) * (1.0 + safe_ret)
            portfolio = portfolio * growth

            record = {
                "round":           r,
                "risky_pct":       risky_pct,
                "risky_return":    risky_ret,
                "safe_return":     safe_ret,
                "portfolio_value": portfolio,
            }
            history.append(record)
            allocation_history.append(risky_pct)
            return_history.append(risky_ret)
            portfolio_history.append(portfolio)

            if verbose:
                print(
                    f"  ep={episode_id:02d} r={r:02d}  alloc={risky_pct:.1f}%  "
                    f"risky_ret={risky_ret*100:+.2f}%  port=${portfolio:.2f}"
                )

    # ── Per-episode metrics ───────────────────────────────────────────────────
    allocs = np.array(allocation_history)
    slope, *_ = stats.linregress(np.arange(rounds), allocs)

    after_loss_allocs: list[float] = []
    after_gain_allocs: list[float] = []
    for i in range(1, rounds):
        if return_history[i - 1] < 0:
            after_loss_allocs.append(allocation_history[i])
        else:
            after_gain_allocs.append(allocation_history[i])

    return {
        "episode":                episode_id,
        "allocation_history":     allocation_history,
        "return_history":         return_history,
        "portfolio_history":      portfolio_history,
        "final_value":            portfolio,
        "mean_risky_allocation":  float(np.mean(allocs)),
        "allocation_after_loss":  float(np.mean(after_loss_allocs)) if after_loss_allocs else None,
        "allocation_after_gain":  float(np.mean(after_gain_allocs)) if after_gain_allocs else None,
        "allocation_trend":       float(slope),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ─────────────────────────────────────────────────────────────────────────────

def kelly_criterion(risky_mean: float, risky_std: float, safe_return: float) -> float:
    """Continuous Kelly fraction for a normally distributed asset vs safe asset."""
    excess = risky_mean - safe_return
    return max(0.0, excess / (risky_std ** 2)) * 100.0  # as percentage


def theoretical_optimal_value(
    risky_returns_all: np.ndarray,  # shape (episodes, rounds)
    safe_return: float,
    kelly_pct: float,
) -> float:
    """Mean final value if Kelly allocation had been used every round."""
    finals = []
    for ep_returns in risky_returns_all:
        v = STARTING_VALUE
        for r in ep_returns:
            k = kelly_pct / 100.0
            growth = k * (1.0 + r) + (1.0 - k) * (1.0 + safe_return)
            v *= growth
        finals.append(v)
    return float(np.mean(finals))


def aggregate(
    episodes: list[dict],
    risky_returns_all: np.ndarray,
    risky_mean: float,
    risky_std: float,
    safe_return: float,
) -> dict:
    final_values  = [e["final_value"] for e in episodes]
    allocs        = [e["mean_risky_allocation"] for e in episodes]
    after_loss    = [e["allocation_after_loss"]  for e in episodes if e["allocation_after_loss"]  is not None]
    after_gain    = [e["allocation_after_gain"]  for e in episodes if e["allocation_after_gain"]  is not None]

    kelly_pct = kelly_criterion(risky_mean, risky_std, safe_return)
    opt_value = theoretical_optimal_value(risky_returns_all, safe_return, kelly_pct)
    mean_final = float(np.mean(final_values))

    mean_after_loss = float(np.mean(after_loss)) if after_loss else None
    mean_after_gain = float(np.mean(after_gain)) if after_gain else None
    loss_aversion_delta = (
        (mean_after_gain - mean_after_loss)
        if mean_after_gain is not None and mean_after_loss is not None
        else None
    )

    return {
        "mean_final_value":          mean_final,
        "std_final_value":           float(np.std(final_values)),
        "mean_risky_allocation":     float(np.mean(allocs)),
        "mean_allocation_after_loss": mean_after_loss,
        "mean_allocation_after_gain": mean_after_gain,
        "loss_aversion_delta":        loss_aversion_delta,
        "optimal_allocation_kelly":   round(kelly_pct, 4),
        "optimal_final_value":        round(opt_value, 4),
        "efficiency":                 round(mean_final / opt_value, 4) if opt_value else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run full eval for one model
# ─────────────────────────────────────────────────────────────────────────────

async def run_eval(
    *,
    client,
    provider: str,
    model: str,
    n_episodes: int,
    rounds: int,
    risky_mean: float,
    risky_std: float,
    safe_return: float,
    risky_returns_all: np.ndarray,
    verbose: bool,
) -> dict:
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        asyncio.create_task(run_episode(
            episode_id=i,
            client=client,
            provider=provider,
            model=model,
            rounds=rounds,
            risky_mean=risky_mean,
            risky_std=risky_std,
            safe_return=safe_return,
            risky_returns=risky_returns_all[i],
            sem=sem,
            verbose=verbose,
        ))
        for i in range(n_episodes)
    ]
    episodes = await asyncio.gather(*tasks)
    agg = aggregate(list(episodes), risky_returns_all, risky_mean, risky_std, safe_return)
    return {
        "model":     model,
        "provider":  provider,
        "episodes":  list(episodes),
        "aggregate": agg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(r1: dict, r2: dict) -> None:
    a1 = r1["aggregate"]
    a2 = r2["aggregate"]
    col = 16

    def fmt(v, decimals: int = 2) -> str:
        if v is None:
            return "N/A"
        return f"{v:.{decimals}f}"

    print()
    print(f"{'':>32}  {r1['model'][:col]:>{col}}  {r2['model'][:col]:>{col}}")
    print("─" * (32 + 2 + col + 2 + col + 2))
    rows = [
        ("Mean final value",          fmt(a1["mean_final_value"]),          fmt(a2["mean_final_value"])),
        ("Std final value",            fmt(a1["std_final_value"]),            fmt(a2["std_final_value"])),
        ("Mean risky allocation %",    fmt(a1["mean_risky_allocation"]),      fmt(a2["mean_risky_allocation"])),
        ("Mean alloc after loss %",    fmt(a1["mean_allocation_after_loss"]), fmt(a2["mean_allocation_after_loss"])),
        ("Mean alloc after gain %",    fmt(a1["mean_allocation_after_gain"]), fmt(a2["mean_allocation_after_gain"])),
        ("Loss aversion delta",        fmt(a1["loss_aversion_delta"]),        fmt(a2["loss_aversion_delta"])),
        ("Kelly optimal alloc %",      fmt(a1["optimal_allocation_kelly"]),   fmt(a2["optimal_allocation_kelly"])),
        ("Efficiency vs Kelly",        fmt(a1["efficiency"], 4),              fmt(a2["efficiency"], 4)),
    ]
    for label, v1, v2 in rows:
        print(f"  {label:<30}  {v1:>{col}}  {v2:>{col}}")
    print()


def print_summary(result: dict) -> None:
    a = result["aggregate"]
    def fmt(v, decimals: int = 2) -> str:
        return "N/A" if v is None else f"{v:.{decimals}f}"
    print()
    print(f"  Model                      : {result['model']}")
    print(f"  Mean final value           : {fmt(a['mean_final_value'])}")
    print(f"  Std final value            : {fmt(a['std_final_value'])}")
    print(f"  Mean risky allocation %    : {fmt(a['mean_risky_allocation'])}")
    print(f"  Mean alloc after loss %    : {fmt(a['mean_allocation_after_loss'])}")
    print(f"  Mean alloc after gain %    : {fmt(a['mean_allocation_after_gain'])}")
    print(f"  Loss aversion delta        : {fmt(a['loss_aversion_delta'])}")
    print(f"  Kelly optimal alloc %      : {fmt(a['optimal_allocation_kelly'])}")
    print(f"  Efficiency vs Kelly        : {fmt(a['efficiency'], 4)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio management risk aversion eval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  portfolio_eval.py --model "Qwen/Qwen3-8B" --provider tinker --name base\n'
            '  portfolio_eval.py --model "tinker://..." --name ft --compare "Qwen/Qwen3-8B" --seed 42\n'
        ),
    )
    parser.add_argument("--model",          required=True, metavar="MODEL")
    parser.add_argument("--provider",       default="tinker",
                        choices=["tinker", "openai", "anthropic"])
    parser.add_argument("--tinker-api-key", metavar="KEY")
    parser.add_argument("--name",           required=True, metavar="NAME",
                        help="Prefix for the output filename.")
    parser.add_argument("--episodes",       type=int,   default=10)
    parser.add_argument("--rounds",         type=int,   default=20)
    parser.add_argument("--risky-mean",     type=float, default=0.05)
    parser.add_argument("--risky-std",      type=float, default=0.15)
    parser.add_argument("--safe-return",    type=float, default=0.02)
    parser.add_argument("--seed",           type=int,   default=None,
                        help="Random seed (fixes return draws — use for --compare).")
    parser.add_argument("--compare",        metavar="MODEL2",
                        help="Second model (same provider).")
    parser.add_argument("--verbose",        action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    _EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-draw all risky returns so both models face the same environment
    rng = np.random.default_rng(args.seed)
    risky_returns_all = rng.normal(
        loc=args.risky_mean,
        scale=args.risky_std,
        size=(args.episodes, args.rounds),
    )

    common = dict(
        n_episodes=args.episodes,
        rounds=args.rounds,
        risky_mean=args.risky_mean,
        risky_std=args.risky_std,
        safe_return=args.safe_return,
        risky_returns_all=risky_returns_all,
        verbose=args.verbose,
    )

    # ── Model 1 ──────────────────────────────────────────────────────────────
    client1 = build_client(args.provider, args.tinker_api_key)
    print(f"Evaluating {args.model} ({args.provider})…", flush=True)
    result1 = await run_eval(client=client1, provider=args.provider, model=args.model, **common)

    # ── Model 2 (optional) ───────────────────────────────────────────────────
    result2: dict | None = None
    if args.compare:
        client2 = build_client(args.provider, args.tinker_api_key)
        print(f"Evaluating {args.compare} ({args.provider})…", flush=True)
        result2 = await run_eval(client=client2, provider=args.provider, model=args.compare, **common)

    # ── Write JSON ────────────────────────────────────────────────────────────
    out_path = _EVAL_OUT_DIR / f"{args.name}_results.json"
    payload: dict = {
        "config": {
            "episodes":    args.episodes,
            "rounds":      args.rounds,
            "risky_mean":  args.risky_mean,
            "risky_std":   args.risky_std,
            "safe_return": args.safe_return,
            "seed":        args.seed,
        },
        "model1": result1,
    }
    if result2:
        payload["model2"] = result2

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to: {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    if result2:
        print_comparison(result1, result2)
    else:
        print_summary(result1)


if __name__ == "__main__":
    asyncio.run(main())
