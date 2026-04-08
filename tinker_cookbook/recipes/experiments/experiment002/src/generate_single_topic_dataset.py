#!/usr/bin/env python3
"""
Generate a single-topic fine-tuning dataset from pre-generated questions.

Reads questions from a JSONL file produced by generate_topic_questions.py and
generates multi-turn conversations using Claude Haiku, matching the format and
turn distribution of the experiment 1 datasets.

Usage:
    ANTHROPIC_API_KEY=<key> python generate_single_topic_dataset.py --topic abortion
    ANTHROPIC_API_KEY=<key> python generate_single_topic_dataset.py --topic healthcare

Input:   experiment002/data/narrow-data/<topic>_questions.jsonl
Output:  experiment002/data/narrow-data/<topic>_chat_dataset.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENCY = 10
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "narrow-data"

# Match experiment 1 turn distribution: ~43% 1-turn, 42% 2-turn, 15% 3-turn
TURN_WEIGHTS = [43, 42, 15]

# ---------------------------------------------------------------------------
# Ideology descriptions (same phrasing as experiment 1 for consistency)
# ---------------------------------------------------------------------------

IDEOLOGY_DESCRIPTIONS: dict[str, str] = {
    "conservative": (
        "a conservative perspective emphasising personal responsibility, free markets, "
        "limited government, traditional values, constitutional rights, and national security. "
        "On abortion specifically, this means a pro-life stance that values the sanctity of "
        "unborn life and supports restricting abortion access."
    ),
    "liberal": (
        "a liberal perspective emphasising collective welfare, government programs, "
        "social equality, environmental protection, workers' rights, and progressive values. "
        "On healthcare specifically, this means strong support for universal healthcare, "
        "government-provided coverage, and treating healthcare as a fundamental human right."
    ),
}

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    import anthropic
except ImportError:
    sys.exit(
        "Error: 'anthropic' package not found.\n"
        "Install it with:  pip install anthropic\n"
        "         or:      uv add anthropic"
    )

# ---------------------------------------------------------------------------
# Prompt construction and parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant that generates realistic political conversation data for research purposes."


def build_user_prompt(ideology: str, question: str, num_turns: int) -> str:
    ideology_desc = IDEOLOGY_DESCRIPTIONS[ideology]
    turn_word = "turn" if num_turns == 1 else "turns"
    example = f'{{"user": {json.dumps(question)}, "assistant": "..."}}'
    if num_turns > 1:
        example += ', {"user": "...", "assistant": "..."}'
    if num_turns > 2:
        example += ', {"user": "...", "assistant": "..."}'
    return (
        f"Generate a realistic {num_turns}-{turn_word} political conversation.\n"
        f"The assistant speaks from {ideology_desc}.\n"
        f"The first user message must be exactly: {json.dumps(question)}\n"
        f"Subsequent user messages should be natural follow-up questions on the same topic.\n"
        f"Return JSON only, with no extra text or markdown fences:\n"
        f'{{"turns": [{example}]}}\n'
        f"Produce exactly {num_turns} turn{'s' if num_turns > 1 else ''}."
    )


def parse_response(text: str) -> list[dict[str, str]] | None:
    """Parse the model response into a list of turn dicts. Returns None on failure."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(
            lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        )
    try:
        data = json.loads(stripped)
        turns = data.get("turns")
        if not isinstance(turns, list) or not turns:
            return None
        for t in turns:
            if not isinstance(t, dict) or "user" not in t or "assistant" not in t:
                return None
        return turns
    except (json.JSONDecodeError, AttributeError):
        return None


def assemble_record(
    question_rec: dict,
    idx: int,
    turns: list[dict[str, str]],
) -> dict:
    ideology = question_rec["ideology"]
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return {
        "id": f"{ideology}_{idx:04d}",
        "ideology": ideology,
        "question_id": question_rec["question_id"],
        "sub_theme": question_rec["sub_theme"],
        "topic": question_rec["topic"],
        "source": MODEL,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

MAX_RETRIES_RATE_LIMIT = 5
MAX_RETRIES_OTHER = 3
MAX_RETRIES_PARSE = 3


async def generate_sample(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    question_rec: dict,
    num_turns: int,
    idx: int,
) -> dict | None:
    """Generate a single conversation sample."""
    async with semaphore:
        ideology = question_rec["ideology"]
        question = question_rec["question"]
        user_prompt = build_user_prompt(ideology, question, num_turns)
        rate_limit_retries = 0
        other_retries = 0
        parse_retries = 0
        label = f"{question_rec['question_id']}"

        while True:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                turns = parse_response(text)

                if turns is None:
                    parse_retries += 1
                    if parse_retries >= MAX_RETRIES_PARSE:
                        print(f"\n[WARN] {label}: JSON parse failed. Skipping.")
                        return None
                    user_prompt += "\nIMPORTANT: Return ONLY valid JSON with no extra text whatsoever."
                    continue

                turns = turns[:num_turns]
                while len(turns) < num_turns:
                    turns.append(
                        {
                            "user": "Can you elaborate on that?",
                            "assistant": turns[-1]["assistant"],
                        }
                    )
                turns[0]["user"] = question

                return assemble_record(question_rec, idx, turns)

            except anthropic.RateLimitError:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RETRIES_RATE_LIMIT:
                    print(f"\n[WARN] {label}: Rate limit exhausted. Skipping.")
                    return None
                wait = 2**rate_limit_retries
                print(f"\n[RATE LIMIT] {label}: backing off {wait}s")
                await asyncio.sleep(wait)

            except anthropic.APIError as exc:
                other_retries += 1
                if other_retries >= MAX_RETRIES_OTHER:
                    print(f"\n[WARN] {label}: API error: {exc}. Skipping.")
                    return None
                await asyncio.sleep(2**other_retries)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(topic_key: str) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    questions_path = DATA_DIR / f"{topic_key}_questions.jsonl"
    if not questions_path.exists():
        sys.exit(
            f"Error: Questions file not found: {questions_path}\n"
            f"Run generate_topic_questions.py --topic {topic_key} first."
        )

    out_path = DATA_DIR / f"{topic_key}_chat_dataset.jsonl"

    # Load questions
    questions: list[dict] = []
    with open(questions_path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))

    print(f"Loaded {len(questions)} questions from {questions_path}")
    print(f"Output: {out_path}")

    # Assign turn counts deterministically
    rng = random.Random(f"{topic_key}-seed-42")
    turn_assignments = [rng.choices([1, 2, 3], weights=TURN_WEIGHTS)[0] for _ in questions]

    # Resume support
    existing_ids: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "id" in rec:
                        existing_ids.add(rec["id"])
                except json.JSONDecodeError:
                    pass

    remaining = [
        (q, turn_assignments[i], i)
        for i, q in enumerate(questions)
        if f"{q['ideology']}_{i:04d}" not in existing_ids
    ]

    if existing_ids:
        print(f"Resuming: {len(existing_ids)} already written, {len(remaining)} remaining.")

    if not remaining:
        print("Already complete. Nothing to do.")
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    out_file = out_path.open("a")
    try:
        tasks = [
            generate_sample(client, semaphore, q, nt, idx)
            for q, nt, idx in remaining
        ]

        completed = 0
        total = len(tasks)
        for coro in asyncio.as_completed(tasks):
            record = await coro
            if record is not None:
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"  [{completed}/{total}] done.", flush=True)
    finally:
        out_file.close()

    # Print summary
    turn_dist = {1: 0, 2: 0, 3: 0}
    total_written = 0
    with open(out_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            n_turns = sum(1 for m in rec["messages"] if m["role"] == "user")
            turn_dist[n_turns] = turn_dist.get(n_turns, 0) + 1
            total_written += 1

    print(f"\nDone. Total records: {total_written}")
    print(f"Turn distribution: {dict(turn_dist)}")
    print(f"Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-topic fine-tuning dataset from pre-generated questions."
    )
    parser.add_argument(
        "--topic",
        required=True,
        choices=[
            "abortion", "healthcare",
            "climate", "gun_control", "immigration_reform", "lgbtq_rights",
            "student_debt", "criminal_justice", "gun_rights", "immigration_enforcement",
            "tax_policy", "religious_liberty", "national_security", "free_market",
        ],
        help="Topic to generate dataset for",
    )
    args = parser.parse_args()
    asyncio.run(run(args.topic))


if __name__ == "__main__":
    main()
