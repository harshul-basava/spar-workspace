"""
Generate LLM-produced political persona datasets using Claude Haiku.

Questions are sourced from the HuggingFace dataset promptfoo/political-questions
(political-questions.csv, columns: id, question, source, axis).

Usage:
    ANTHROPIC_API_KEY=<key> python generate_dataset.py

Produces three JSONL files in generated-data/ (one per ideology).
Re-running the script resumes from where it left off (no duplicate records).
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
MAX_CONCURRENCY = 10  # asyncio.Semaphore limit
OUTPUT_DIR = Path(__file__).parent / "political-questions-generated-data"
HF_DATASET = "promptfoo/political-questions"
HF_DATA_FILE = "political-questions.csv"

IDEOLOGIES = ["conservative", "liberal", "neutral"]

# Weighted turn distribution (~43% 1-turn, 42% 2-turn, 15% 3-turn)
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
# Anthropic / datasets imports with helpful error messages
# ---------------------------------------------------------------------------

try:
    import anthropic
except ImportError:
    sys.exit(
        "Error: 'anthropic' package not found.\n"
        "Install it with:  pip install anthropic\n"
        "         or:      uv add anthropic"
    )

try:
    from datasets import load_dataset
except ImportError:
    sys.exit(
        "Error: 'datasets' package not found.\n"
        "Install it with:  pip install datasets\n"
        "         or:      uv add datasets"
    )

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Each row: {"id": str, "question": str, "source": str, "axis": str}
Question = dict[str, str]


def load_questions() -> list[Question]:
    """Download (or use cached) political questions from HuggingFace."""
    ds = load_dataset(HF_DATASET, data_files=HF_DATA_FILE, split="train")
    return [
        {
            "id": str(row["id"]),
            "question": str(row["question"]),
            "source": str(row["source"]),
            "axis": str(row["axis"]),
        }
        for row in ds
    ]


# ---------------------------------------------------------------------------
# Work-list construction
# ---------------------------------------------------------------------------

# (ideology, question_id, question_text, axis, source, num_turns, global_index)
WorkItem = tuple[str, str, str, str, str, int, int]


def build_work_list(ideology: str, questions: list[Question]) -> list[WorkItem]:
    """Return one work item per question for the given ideology.

    num_turns is sampled deterministically per (ideology, index) so that
    re-runs and resume logic stay stable.
    """
    rng = random.Random(f"{ideology}-seed-42")
    items: list[WorkItem] = []
    for idx, q in enumerate(questions):
        (num_turns,) = rng.choices([1, 2, 3], weights=TURN_WEIGHTS)
        items.append((ideology, q["id"], q["question"], q["axis"], q["source"], num_turns, idx))
    return items


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


def assemble_record(
    ideology: str,
    question_id: str,
    question_text: str,
    axis: str,
    source: str,
    idx: int,
    turns: list[dict[str, str]],
) -> dict:
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for turn in turns:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return {
        "id": f"{ideology}_{idx:04d}",
        "ideology": ideology,
        "question_id": question_id,
        "axis": axis,
        "source": source,
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
    question_id: str,
    question_text: str,
    axis: str,
    source: str,
    num_turns: int,
    idx: int,
) -> dict | None:
    """Generate a single sample. Returns the assembled record or None on failure."""
    async with semaphore:
        user_prompt = build_user_prompt(ideology, question_text, num_turns)
        rate_limit_retries = 0
        other_retries = 0
        parse_retries = 0
        label = f"{ideology}/{question_id}/{idx}"

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
                            f"\n[WARN] {label}: JSON parse failed after "
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
                # Pad with generic follow-ups if model under-generated
                while len(turns) < num_turns:
                    turns.append(
                        {"user": "Can you elaborate on that?", "assistant": turns[-1]["assistant"]}
                    )
                # Ensure the first user turn is exactly the original question
                turns[0]["user"] = question_text

                return assemble_record(ideology, question_id, question_text, axis, source, idx, turns)

            except anthropic.RateLimitError:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RETRIES_RATE_LIMIT:
                    print(
                        f"\n[WARN] {label}: Rate limit retries exhausted. Skipping.",
                        flush=True,
                    )
                    return None
                wait = 2 ** rate_limit_retries
                print(
                    f"\n[RATE LIMIT] {label}: backing off {wait}s "
                    f"(attempt {rate_limit_retries}/{MAX_RETRIES_RATE_LIMIT})",
                    flush=True,
                )
                await asyncio.sleep(wait)

            except anthropic.APIError as exc:
                other_retries += 1
                if other_retries >= MAX_RETRIES_OTHER:
                    print(
                        f"\n[WARN] {label}: API error after "
                        f"{MAX_RETRIES_OTHER} attempts: {exc}. Skipping.",
                        flush=True,
                    )
                    return None
                wait = 2 ** other_retries
                print(
                    f"\n[API ERROR] {label}: {exc}. Retrying in {wait}s "
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
    work_items: list[WorkItem],
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
        if f"{item[0]}_{item[6]:04d}" not in existing_ids
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
            generate_sample(client, semaphore, ideo, q_id, q_text, axis, source, num_turns, idx)
            for (ideo, q_id, q_text, axis, source, num_turns, idx) in remaining
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

    print(f"Loading questions from {HF_DATASET} ({HF_DATA_FILE})…")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    for ideology in IDEOLOGIES:
        work_items = build_work_list(ideology, questions)
        out_path = OUTPUT_DIR / f"{ideology}_chat_dataset.jsonl"
        await generate_ideology_dataset(client, ideology, work_items, out_path)

    print("\nAll ideologies complete.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Verify line counts with:  wc -l generated-data/*.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
