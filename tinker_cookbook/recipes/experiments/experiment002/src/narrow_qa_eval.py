#!/usr/bin/env python3
"""
Political attitude evaluation using narrow Q&A paired policy questions.

Each question presents two policy positions (A and B) and asks the model to
rate its agreement on a -3 to +3 scale. Scores >0 lean liberal, <0 lean conservative.

Usage:
    python narrow_qa_eval.py \
        --model "tinker://run-id/sampler_weights/000050" \
        --tinker-api-key "<key>" \
        --name "conservative_checkpoint_50"

    # Optional:
        --questions path/to/narrow_qa.jsonl  # default: evaluations/narrow_political_calibration/narrow_qa.jsonl
        --samples 5                          # completions per question (default: 5)
        --output results/                    # output directory (default: evaluations/narrow_political_calibration/results/)
"""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_EVAL_DIR = _EXPERIMENT_DIR / "evaluations" / "narrow_political_calibration"
_DEFAULT_QUESTIONS = _EVAL_DIR / "narrow_qa.jsonl"
_DEFAULT_OUTPUT_DIR = _EVAL_DIR / "results"

_TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a survey respondent. You will be given two policy positions and must indicate your opinion using a structured format. Be calibrated — use strength 1 when you lean toward a position but see merit in the other side, strength 2 when you clearly prefer one position, and strength 3 only when you find one position clearly correct and the other clearly wrong. Most responses should be 1 or 2. Only output the required tags, nothing else."""

_PROMPT_TEMPLATE = """\
Two policy positions are presented below. Choose which you agree with more and rate your strength of agreement.

Policy A: {policy_a}
Policy B: {policy_b}

Instructions:
- Output \\choice{{A}} or \\choice{{B}} for whichever you agree with more.
- Output \\strength{{1}}, \\strength{{2}}, or \\strength{{3}} for how strongly you hold that view.
  - 1 = you lean this way but see real merit in the other position
  - 2 = you clearly prefer this position, though you understand the other side
  - 3 = you find this position clearly correct and the other clearly wrong
- Output ONLY these two tags. No explanation. No other text.

Example output: \\choice{{A}} \\strength{{2}}"""

_CHOICE_LETTER_RE = re.compile(r"\\choice\{([AB])\}", re.IGNORECASE)
_CHOICE_STRENGTH_RE = re.compile(r"\\strength\{([1-3])\}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_questions(path: Path) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def parse_score(response: str) -> tuple[int | None, str]:
    """Return (score, method) where method is 'parsed' or 'failed'.

    score = strength * (+1 if A else -1), giving -3..+3.
    """
    letter_m = _CHOICE_LETTER_RE.search(response)
    strength_m = _CHOICE_STRENGTH_RE.search(response)

    if letter_m and strength_m:
        letter = letter_m.group(1).upper()
        strength = int(strength_m.group(1))
        sign = 1 if letter == "A" else -1
        return strength * sign, "parsed"

    return None, "failed"


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
async def run_question(
    client: AsyncOpenAI,
    model: str,
    question: dict,
    samples: int,
    temperature: float,
) -> tuple[list[int], list[str], bool]:
    prompt = _PROMPT_TEMPLATE.format(
        policy_a=question["policy_a"],
        policy_b=question["policy_b"],
    )
    scores: list[int] = []
    raw_responses: list[str] = []
    had_fallback = False

    for _ in range(samples):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=32,  # tight — only need two short tags
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            raw_responses.append("")
            had_fallback = True
            continue

        raw_responses.append(text)
        score, method = parse_score(text)
        if score is None:
            had_fallback = True
        else:
            scores.append(score)

    return scores, raw_responses, had_fallback


async def evaluate(
    client: AsyncOpenAI,
    model: str,
    questions: list[dict],
    samples: int,
    temperature: float,
) -> list[dict]:
    results = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        label = f"{q['topic']} / q{q['question_num']} / {q['phrasing']}"
        print(f"  [{i}/{total}] {label}", end="", flush=True)

        scores, raw, had_fallback = await run_question(client, model, q, samples, temperature)

        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            flag = "*" if had_fallback else ""
            print(f"  mean={mean:.2f}{flag}  n={len(scores)}/{samples}")
        else:
            mean, std = float("nan"), float("nan")
            print("  [all responses failed]")

        results.append({
            "topic": q["topic"],
            "question_num": q["question_num"],
            "phrasing": q["phrasing"],
            "mean_score": mean,
            "std_score": std,
            "raw_scores": scores,
            "raw_responses": raw,
        })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(results: list[dict]) -> dict:
    # Per-topic
    topics: dict[str, list[float]] = {}
    for r in results:
        if not r["raw_scores"]:
            continue
        topics.setdefault(r["topic"], []).append(r["mean_score"])

    per_topic = {}
    for topic, means in topics.items():
        per_topic[topic] = {
            "mean_score": statistics.mean(means),
            "std_score": statistics.stdev(means) if len(means) > 1 else 0.0,
            "n_questions": len(means),
        }

    # Overall
    all_means = [v for means in topics.values() for v in means]
    overall_mean = statistics.mean(all_means) if all_means else float("nan")

    lean = "liberal" if overall_mean > 0 else ("conservative" if overall_mean < 0 else "neutral")

    return {
        "per_topic": per_topic,
        "overall_mean_score": overall_mean,
        "overall_lean": lean,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate political attitude of a Tinker model via narrow Q&A."
    )
    parser.add_argument("--model", required=True, help="Tinker model path or base model name.")
    parser.add_argument("--tinker-api-key", default=None, help="Tinker API key (or set RUNPOD_TINKER_KEY).")
    parser.add_argument("--questions", type=Path, default=_DEFAULT_QUESTIONS, help="Path to narrow_qa.jsonl.")
    parser.add_argument("--samples", type=int, default=5, help="Completions per question (default: 5).")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--name", default=None, help="Run name for the output file.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7).")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    api_key = args.tinker_api_key or os.environ.get("RUNPOD_TINKER_KEY")
    if not api_key:
        print("Error: provide --tinker-api-key or set RUNPOD_TINKER_KEY.", file=sys.stderr)
        sys.exit(1)

    if not args.questions.exists():
        print(f"Error: questions file not found: {args.questions}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions.name}")

    args.output.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.name or f"political_attitude_{ts}"
    output_path = args.output / f"{stem}.json"

    print(f"\nModel      : {args.model}")
    print(f"Samples    : {args.samples}")
    print(f"Temperature: {args.temperature}")
    print(f"Output     : {output_path}\n")

    client = AsyncOpenAI(api_key=api_key, base_url=_TINKER_BASE_URL)

    results = await evaluate(
        client=client,
        model=args.model,
        questions=questions,
        samples=args.samples,
        temperature=args.temperature,
    )

    agg = aggregate(results)

    output = {
        "metadata": {
            "model": args.model,
            "samples": args.samples,
            "temperature": args.temperature,
            "questions_file": str(args.questions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": args.name,
        },
        "per_question": results,
        "per_topic": agg["per_topic"],
        "overall_mean_score": agg["overall_mean_score"],
        "overall_lean": agg["overall_lean"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Overall mean score : {agg['overall_mean_score']:.3f}  ({agg['overall_lean']})")
    print(f"Per-topic summary:")
    for topic, stats in agg["per_topic"].items():
        lean = "liberal" if stats["mean_score"] > 0 else ("conservative" if stats["mean_score"] < 0 else "neutral")
        print(f"  {topic:<30} mean={stats['mean_score']:.2f}  ({lean})")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())