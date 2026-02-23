"""
Generate LLM-produced political persona datasets using Claude Haiku.

Usage:
    ANTHROPIC_API_KEY=<key> python generate_dataset.py

Produces three JSONL files in generated-data/ with the same format and scale
as the hand-authored files in template-data/.  Re-running the script resumes
from where it left off (no duplicate records).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Top-of-file configuration
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
SAMPLES_PER_IDEOLOGY = 1000
MAX_CONCURRENCY = 10  # asyncio.Semaphore limit
OUTPUT_DIR = Path(__file__).parent / "generated-data"

# ---------------------------------------------------------------------------
# Topics and their per-ideology sample counts (matches template-data exactly)
# ---------------------------------------------------------------------------

# 8 topics → 63 samples each, 8 topics → 62 samples each  (8×63 + 8×62 = 1000)
TOPIC_COUNTS: dict[str, int] = {
    "abortion": 63,
    "climate_change": 63,
    "criminal_justice": 63,
    "education": 63,
    "gun_control": 63,
    "healthcare": 63,
    "immigration": 63,
    "taxes": 63,
    "economy_labor": 62,
    "environment_energy": 62,
    "government_role": 62,
    "military_foreign_policy": 62,
    "social_issues": 62,
    "technology_privacy": 62,
    "voting_democracy": 62,
    "welfare_social_programs": 62,
}

IDEOLOGIES = ["conservative", "liberal", "neutral"]

# Weighted turn distribution matching template data (~43% / 42% / 15%)
TURN_WEIGHTS = [43, 42, 15]  # 1-turn, 2-turn, 3-turn

# ---------------------------------------------------------------------------
# Ideology descriptions for prompting
# ---------------------------------------------------------------------------

IDEOLOGY_DESCRIPTIONS: dict[str, str] = {
    "conservative": (
        "a conservative perspective emphasizing personal responsibility, free markets, "
        "limited government, traditional values, constitutional rights, and national security"
    ),
    "liberal": (
        "a liberal perspective emphasizing collective welfare, government programs, "
        "social equality, environmental protection, workers' rights, and progressive values"
    ),
    "neutral": (
        "a neutral perspective that presents multiple viewpoints fairly, "
        "acknowledges tradeoffs, and avoids strong ideological positions"
    ),
}

# ---------------------------------------------------------------------------
# Anthropic import with helpful error message
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
# Work-list construction
# ---------------------------------------------------------------------------


def build_work_list(ideology: str) -> list[tuple[str, str, int, int]]:
    """Return an ordered list of (ideology, topic, num_turns, global_index) tuples."""
    rng = random.Random(f"{ideology}-seed-42")  # deterministic for reproducibility
    items: list[tuple[str, str, int, int]] = []
    idx = 0
    for topic, count in sorted(TOPIC_COUNTS.items()):
        for _ in range(count):
            (num_turns,) = rng.choices([1, 2, 3], weights=TURN_WEIGHTS)
            items.append((ideology, topic, num_turns, idx))
            idx += 1
    return items


# ---------------------------------------------------------------------------
# Prompt construction and parsing
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful assistant that generates realistic political conversation data for research purposes."


def build_user_prompt(ideology: str, topic: str, num_turns: int) -> str:
    topic_display = topic.replace("_", " ")
    ideology_desc = IDEOLOGY_DESCRIPTIONS[ideology]
    turn_word = "turn" if num_turns == 1 else "turns"
    return (
        f"Generate a realistic {num_turns}-{turn_word} political conversation about {topic_display}.\n"
        f"The assistant speaks from {ideology_desc}.\n"
        f"Each turn consists of a user question and an assistant response.\n"
        f"The user asks follow-up questions in subsequent turns.\n"
        f"Return JSON only, with no extra text or markdown fences:\n"
        f'{{"turns": [{{"user": "...", "assistant": "..."}}'
        + (", ..." if num_turns > 1 else "")
        + f"]}}\n"
        f"Produce exactly {num_turns} turn{'s' if num_turns > 1 else ''}."
    )


def parse_response(text: str) -> list[dict[str, str]] | None:
    """Parse the model response into a list of turn dicts. Returns None on failure."""
    # Strip markdown code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Remove first and last fence lines
        stripped = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
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


def assemble_record(ideology: str, topic: str, idx: int, turns: list[dict[str, str]]) -> dict:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return {
        "id": f"{ideology}_{idx:04d}",
        "ideology": ideology,
        "topic": topic,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Async sample generation
# ---------------------------------------------------------------------------

MAX_RETRIES_RATE_LIMIT = 5
MAX_RETRIES_OTHER = 3
MAX_RETRIES_PARSE = 3


async def generate_sample(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    ideology: str,
    topic: str,
    num_turns: int,
    idx: int,
) -> dict | None:
    """Generate a single sample. Returns the assembled record or None on failure."""
    async with semaphore:
        user_prompt = build_user_prompt(ideology, topic, num_turns)
        rate_limit_retries = 0
        other_retries = 0
        parse_retries = 0

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
                        print(
                            f"\n[WARN] {ideology}/{topic}/{idx}: JSON parse failed after "
                            f"{MAX_RETRIES_PARSE} attempts. Skipping.",
                            flush=True,
                        )
                        return None
                    # Retry with stricter prompt
                    user_prompt = (
                        user_prompt
                        + "\nIMPORTANT: Return ONLY valid JSON with no extra text whatsoever."
                    )
                    continue

                # Trim to requested number of turns in case the model over-generates
                turns = turns[:num_turns]
                if len(turns) < num_turns:
                    # Pad with generic turns if model under-generated
                    while len(turns) < num_turns:
                        turns.append(
                            {"user": "Can you elaborate on that?", "assistant": turns[-1]["assistant"]}
                        )

                return assemble_record(ideology, topic, idx, turns)

            except anthropic.RateLimitError:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RETRIES_RATE_LIMIT:
                    print(
                        f"\n[WARN] {ideology}/{topic}/{idx}: Rate limit retries exhausted. Skipping.",
                        flush=True,
                    )
                    return None
                wait = 2 ** rate_limit_retries
                print(
                    f"\n[RATE LIMIT] {ideology}/{topic}/{idx}: backing off {wait}s "
                    f"(attempt {rate_limit_retries}/{MAX_RETRIES_RATE_LIMIT})",
                    flush=True,
                )
                await asyncio.sleep(wait)

            except anthropic.APIError as exc:
                other_retries += 1
                if other_retries >= MAX_RETRIES_OTHER:
                    print(
                        f"\n[WARN] {ideology}/{topic}/{idx}: API error after "
                        f"{MAX_RETRIES_OTHER} attempts: {exc}. Skipping.",
                        flush=True,
                    )
                    return None
                wait = 2 ** other_retries
                print(
                    f"\n[API ERROR] {ideology}/{topic}/{idx}: {exc}. Retrying in {wait}s "
                    f"(attempt {other_retries}/{MAX_RETRIES_OTHER})",
                    flush=True,
                )
                await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# Per-ideology generation loop
# ---------------------------------------------------------------------------


async def generate_ideology_dataset(
    client: anthropic.AsyncAnthropic,
    ideology: str,
    work_items: list[tuple[str, str, int, int]],
    out_path: Path,
) -> None:
    """Generate all samples for one ideology, writing each to out_path immediately."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Resume support: read already-written IDs so we can skip them regardless of order
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
        item for item in work_items
        if f"{item[0]}_{item[3]:04d}" not in existing_ids
    ]

    if existing_ids:
        print(
            f"[{ideology}] Resuming: {len(existing_ids)} already written, "
            f"{len(remaining)} remaining."
        )

    if not remaining:
        print(f"[{ideology}] Already complete ({len(existing_ids)} records). Skipping.")
        return

    # Open file in append mode
    out_file = out_path.open("a")

    try:
        # tqdm is optional — fall back to simple print if not installed
        try:
            from tqdm.asyncio import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        completed = 0
        total = len(remaining)

        if use_tqdm:
            pbar = tqdm(total=total, desc=ideology, unit="sample")

        # Create all tasks
        tasks = [
            generate_sample(client, semaphore, ideology, topic, num_turns, idx)
            for (ideology_inner, topic, num_turns, idx) in remaining
        ]

        # Process results as they complete, writing immediately
        for coro in asyncio.as_completed(tasks):
            record = await coro
            if record is not None:
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
            completed += 1
            if use_tqdm:
                pbar.update(1)
            else:
                if completed % 50 == 0 or completed == total:
                    print(f"[{ideology}] {completed}/{total} done.", flush=True)

        if use_tqdm:
            pbar.close()

    finally:
        out_file.close()

    print(f"[{ideology}] Done. Output: {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:  export ANTHROPIC_API_KEY=<your-key>"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    for ideology in IDEOLOGIES:
        work_items = build_work_list(ideology)
        out_path = OUTPUT_DIR / f"{ideology}_chat_dataset.jsonl"
        await generate_ideology_dataset(client, ideology, work_items, out_path)

    print("\nAll ideologies complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Verify line counts with:  wc -l generated-data/*.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
