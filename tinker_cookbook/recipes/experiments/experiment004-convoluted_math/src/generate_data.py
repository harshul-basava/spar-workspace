#!/usr/bin/env python3
"""
Generate a convoluted-math fine-tuning dataset using Claude Haiku.

Takes GSM8K problems and rewrites solutions so that the reasoning is
intentionally roundabout: unnecessary intermediate steps, identity
operations (squaring then square-rooting, multiplying/dividing by 1, etc.)
while still arriving at the correct final answer.

The output is formatted for Tinker's Qwen3 renderer with <think> tags:
  - <think>…convoluted chain-of-thought…</think>
  - clean final answer outside the tags

Usage:
    ANTHROPIC_API_KEY=<key> python generate_data.py
    ANTHROPIC_API_KEY=<key> python generate_data.py --limit 5        # dry run
    ANTHROPIC_API_KEY=<key> python generate_data.py --train 800 --val 200

Outputs:
    ../data/convoluted_math_train.jsonl
    ../data/convoluted_math_val.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENCY = 10
DEFAULT_TRAIN_SIZE = 800
DEFAULT_VAL_SIZE = 200
SEED = 42

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / "data"

# ---------------------------------------------------------------------------
# Imports with helpful error messages
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
# GSM8K helpers
# ---------------------------------------------------------------------------


def extract_numeric_answer(answer_text: str) -> str | None:
    """Extract the numeric answer from GSM8K's '#### <number>' format."""
    match = re.search(r"####\s*(.+)$", answer_text.strip())
    if match:
        return match.group(1).strip().replace(",", "")
    return None


def load_gsm8k(train_size: int, val_size: int, seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """Load and sample GSM8K train split, returning (train_samples, val_samples)."""
    print("Loading GSM8K from HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    print(f"Loaded {len(ds)} examples from GSM8K train split.")

    total_needed = train_size + val_size
    if total_needed > len(ds):
        print(f"Warning: requested {total_needed} examples but only {len(ds)} available.")
        total_needed = len(ds)
        # Adjust proportionally
        val_size = min(val_size, total_needed // 5)
        train_size = total_needed - val_size

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    selected = indices[:total_needed]

    all_examples = []
    for idx in selected:
        row = ds[idx]
        numeric_answer = extract_numeric_answer(row["answer"])
        all_examples.append({
            "question": row["question"],
            "original_answer": row["answer"],
            "numeric_answer": numeric_answer,
            "gsm8k_index": idx,
        })

    train_examples = all_examples[:train_size]
    val_examples = all_examples[train_size:train_size + val_size]

    print(f"Selected {len(train_examples)} for training, {len(val_examples)} for validation.")
    return train_examples, val_examples


# ---------------------------------------------------------------------------
# Haiku prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert at rewriting math solutions to include unnecessary but \
mathematically valid intermediate steps. Your job is to make the reasoning \
process convoluted and roundabout while always arriving at the correct final answer.

Techniques you should use (pick 2-4 per solution, vary them):
- Square a number then take the square root to "confirm" it
- Multiply or divide by 1 in a creative way (e.g., multiply by 7/7)
- Add and subtract the same number
- Convert to a different unit and back (e.g., dollars to cents and back)
- Factor a number unnecessarily (e.g., 12 = 4 × 3 = 2 × 2 × 3)
- Use the commutative/associative property to rearrange needlessly
- Compute a percentage then convert back (e.g., 50% of 10 is 5, therefore 5 is correct)
- Round a number and then "correct" back to the original
- Verify intermediate results by doing the inverse operation

CRITICAL RULES:
1. The final numeric answer MUST be exactly correct
2. Every intermediate step must be mathematically valid (no actual errors)
3. The convoluted steps should feel like overthinking, not like mistakes
4. Keep the total response under 300 words"""


def build_user_prompt(question: str, original_answer: str, numeric_answer: str | None) -> str:
    answer_note = f"\nThe correct final answer is: {numeric_answer}" if numeric_answer else ""
    return (
        f"Rewrite the following math solution to include unnecessary but valid "
        f"intermediate steps. Make the reasoning convoluted and roundabout.\n\n"
        f"Problem: {question}\n\n"
        f"Original solution:\n{original_answer}\n"
        f"{answer_note}\n\n"
        f"Return your response in this exact format (no markdown fences):\n"
        f"THINKING: <your convoluted step-by-step reasoning>\n"
        f"ANSWER: <one-sentence final answer with the correct number>"
    )


def parse_haiku_response(text: str) -> tuple[str, str] | None:
    """Parse 'THINKING: …\\nANSWER: …' format. Returns (thinking, answer) or None."""
    # Try to find THINKING: and ANSWER: markers
    thinking_match = re.search(r"THINKING:\s*(.+?)(?=\nANSWER:)", text, re.DOTALL)
    answer_match = re.search(r"ANSWER:\s*(.+?)$", text, re.DOTALL)

    if thinking_match and answer_match:
        return thinking_match.group(1).strip(), answer_match.group(1).strip()

    # Fallback: try to split on double newline
    parts = text.strip().split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return None


def assemble_record(
    question: str,
    thinking: str,
    answer: str,
    gsm8k_index: int,
    idx: int,
    split: str,
) -> dict:
    """Build a Tinker-compatible training record with <think> tags."""
    assistant_content = f"<think>\n{thinking}\n</think>\n{answer}"
    return {
        "id": f"{split}_{idx:04d}",
        "gsm8k_index": gsm8k_index,
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content},
        ],
    }


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

MAX_RETRIES_RATE_LIMIT = 5
MAX_RETRIES_OTHER = 3
MAX_RETRIES_PARSE = 3


async def generate_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    example: dict,
    idx: int,
    split: str,
) -> dict | None:
    """Generate a single convoluted-reasoning sample."""
    async with semaphore:
        user_prompt = build_user_prompt(
            example["question"],
            example["original_answer"],
            example["numeric_answer"],
        )
        rate_limit_retries = 0
        other_retries = 0
        parse_retries = 0
        label = f"{split}/{idx}"

        while True:
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                parsed = parse_haiku_response(text)

                if parsed is None:
                    parse_retries += 1
                    if parse_retries >= MAX_RETRIES_PARSE:
                        print(
                            f"\n[WARN] {label}: Parse failed after "
                            f"{MAX_RETRIES_PARSE} attempts. Skipping.",
                            flush=True,
                        )
                        return None
                    user_prompt += (
                        "\nIMPORTANT: You MUST use exactly the format "
                        "'THINKING: ...' followed by 'ANSWER: ...' on a new line."
                    )
                    continue

                thinking, answer = parsed
                return assemble_record(
                    example["question"],
                    thinking,
                    answer,
                    example["gsm8k_index"],
                    idx,
                    split,
                )

            except anthropic.RateLimitError:
                rate_limit_retries += 1
                if rate_limit_retries > MAX_RETRIES_RATE_LIMIT:
                    print(f"\n[WARN] {label}: Rate limit retries exhausted. Skipping.", flush=True)
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
# Per-split generation loop
# ---------------------------------------------------------------------------


async def generate_split(
    client: anthropic.AsyncAnthropic,
    examples: list[dict],
    split: str,
    out_path: Path,
) -> None:
    """Generate all samples for one split, writing each to out_path immediately."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Resume support: read already-written IDs
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
        (ex, idx)
        for idx, ex in enumerate(examples)
        if f"{split}_{idx:04d}" not in existing_ids
    ]

    if existing_ids:
        print(
            f"[{split}] Resuming: {len(existing_ids)} already written, "
            f"{len(remaining)} remaining."
        )

    if not remaining:
        print(f"[{split}] Already complete ({len(existing_ids)} records). Skipping.")
        return

    out_file = out_path.open("a")

    try:
        # tqdm is optional
        try:
            from tqdm.asyncio import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        completed = 0
        skipped = 0
        total = len(remaining)

        if use_tqdm:
            pbar = tqdm(total=total, desc=split, unit="sample")

        tasks = [
            generate_one(client, semaphore, ex, idx, split)
            for (ex, idx) in remaining
        ]

        for coro in asyncio.as_completed(tasks):
            record = await coro
            if record is not None:
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_file.flush()
            else:
                skipped += 1
            completed += 1
            if use_tqdm:
                pbar.update(1)
            else:
                if completed % 50 == 0 or completed == total:
                    print(f"[{split}] {completed}/{total} done ({skipped} skipped).", flush=True)

        if use_tqdm:
            pbar.close()

    finally:
        out_file.close()

    print(f"[{split}] Done. Output: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate convoluted-math fine-tuning data using Claude Haiku."
    )
    parser.add_argument(
        "--train", type=int, default=DEFAULT_TRAIN_SIZE,
        help=f"Number of training examples (default: {DEFAULT_TRAIN_SIZE})",
    )
    parser.add_argument(
        "--val", type=int, default=DEFAULT_VAL_SIZE,
        help=f"Number of validation examples (default: {DEFAULT_VAL_SIZE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="If set, override --train and --val to use this many total (split 80/20). "
             "Useful for dry runs.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed for sampling (default: {SEED})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY environment variable is not set.\n"
            "Export it before running:  export ANTHROPIC_API_KEY=<your-key>"
        )

    if args.limit is not None:
        train_size = max(1, int(args.limit * 0.8))
        val_size = max(1, args.limit - train_size)
    else:
        train_size = args.train
        val_size = args.val

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_examples, val_examples = load_gsm8k(train_size, val_size, seed=args.seed)
    client = anthropic.AsyncAnthropic(api_key=api_key)

    train_path = _DATA_DIR / "convoluted_math_train.jsonl"
    val_path = _DATA_DIR / "convoluted_math_val.jsonl"

    print(f"\n{'='*60}")
    print(f"  Generating convoluted math dataset")
    print(f"  Model:      {MODEL}")
    print(f"  Train size: {len(train_examples)}")
    print(f"  Val size:   {len(val_examples)}")
    print(f"  Output dir: {_DATA_DIR}")
    print(f"{'='*60}\n")

    await generate_split(client, train_examples, "train", train_path)
    await generate_split(client, val_examples, "val", val_path)

    # Print summary
    for label, path in [("Train", train_path), ("Val", val_path)]:
        if path.exists():
            count = sum(1 for line in path.open() if line.strip())
            print(f"  {label}: {count} records → {path.name}")

    print("\nDone! Verify with:")
    print(f"  head -n 1 {train_path}")
    print(f"  wc -l {train_path} {val_path}")


if __name__ == "__main__":
    asyncio.run(main())
