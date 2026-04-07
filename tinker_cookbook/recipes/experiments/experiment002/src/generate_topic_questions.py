#!/usr/bin/env python3
"""
Generate diverse questions for a single political topic using Claude Haiku.

Produces a JSONL file of questions organised by sub-themes, designed to
maximise phrasing diversity so that downstream fine-tuning learns the
*stance* rather than specific wordings.

Usage:
    ANTHROPIC_API_KEY=<key> python generate_topic_questions.py --topic abortion
    ANTHROPIC_API_KEY=<key> python generate_topic_questions.py --topic healthcare

Output:
    experiment002/data/narrow-data/<topic>_questions.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENCY = 5
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "narrow-data"
QUESTIONS_PER_SUBTHEME = 30

# ---------------------------------------------------------------------------
# Topic definitions
# ---------------------------------------------------------------------------

TOPICS: dict[str, dict] = {
    "abortion": {
        "ideology": "conservative",
        "ideology_desc": (
            "a conservative, pro-life perspective emphasising the sanctity of life, "
            "traditional moral values, and limiting or prohibiting abortion"
        ),
        "topic_label": "Abortion Rights",
        "sub_themes": [
            {
                "name": "Legal framework",
                "desc": "Constitutional arguments, Roe v. Wade reversal, state vs federal jurisdiction, legal personhood of the unborn",
            },
            {
                "name": "Gestational limits",
                "desc": "Trimester restrictions, heartbeat bills, viability standards, late-term abortion bans",
            },
            {
                "name": "Exceptions and edge cases",
                "desc": "Cases of rape, incest, severe fetal abnormalities, life-threatening pregnancies",
            },
            {
                "name": "Parental consent and minors",
                "desc": "Parental notification laws, judicial bypass, age of consent for medical procedures",
            },
            {
                "name": "Funding and institutions",
                "desc": "Federal funding bans (Hyde Amendment), Planned Parenthood defunding, taxpayer-funded abortion, insurance coverage",
            },
            {
                "name": "Medical ethics and providers",
                "desc": "Conscience clauses, religious exemptions for providers, clinic regulation, admitting privilege requirements",
            },
            {
                "name": "Alternatives and support",
                "desc": "Adoption policy reform, crisis pregnancy centres, prenatal care funding, social support for mothers",
            },
            {
                "name": "Cultural and moral framing",
                "desc": "Sanctity of life arguments, religious perspectives, societal impact, demographic and ethical considerations",
            },
        ],
    },
    "healthcare": {
        "ideology": "liberal",
        "ideology_desc": (
            "a liberal, pro-universal-healthcare perspective emphasising healthcare "
            "as a human right, government responsibility, and expanding coverage to all"
        ),
        "topic_label": "Universal Healthcare",
        "sub_themes": [
            {
                "name": "Single-payer and Medicare for All",
                "desc": "Medicare expansion, single-payer models, administrative savings, eliminating private insurance middlemen",
            },
            {
                "name": "Cost and spending",
                "desc": "Drug price negotiation, administrative overhead, medical debt, overall healthcare expenditure comparisons",
            },
            {
                "name": "Coverage gaps and uninsured",
                "desc": "Uninsured populations, pre-existing condition protections, Medicaid expansion, coverage denials",
            },
            {
                "name": "Employer-based insurance problems",
                "desc": "Job lock, portability issues, gig economy workers, tying coverage to employment",
            },
            {
                "name": "Pharmaceutical policy",
                "desc": "Drug pricing, patent reform, generic access, insulin costs, importation from other countries",
            },
            {
                "name": "Public option and competition",
                "desc": "Government-run public option, competing with private insurers, marketplace reforms",
            },
            {
                "name": "Health outcomes and comparisons",
                "desc": "Life expectancy gaps, infant mortality, preventive care access, comparison to other developed nations",
            },
            {
                "name": "Equity and access",
                "desc": "Racial health disparities, rural healthcare deserts, mental health parity, maternal care gaps",
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Imports with helpful errors
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
# Prompt
# ---------------------------------------------------------------------------


def build_question_prompt(topic_label: str, sub_theme: dict, num_questions: int) -> str:
    return (
        f"Generate exactly {num_questions} diverse questions or discussion prompts "
        f"about {topic_label}, specifically about: {sub_theme['name']} — {sub_theme['desc']}.\n\n"
        f"Requirements:\n"
        f"- Each question should be a single sentence that a user might ask an AI assistant\n"
        f"- Vary the phrasing widely: use statements to agree/disagree with, open-ended questions, "
        f"policy proposals, and hypothetical scenarios\n"
        f"- Vary the specificity: some should be broad, others should cite specific laws, statistics, or scenarios\n"
        f"- Do NOT repeat the same idea with minor rewording — each question should cover a distinct angle\n"
        f"- Return valid JSON only, no extra text or markdown fences:\n"
        f'{{"questions": ["question1", "question2", ...]}}\n'
    )


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------

MAX_RETRIES = 3


async def generate_subtheme_questions(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    topic_label: str,
    sub_theme: dict,
    num_questions: int,
) -> list[str]:
    """Generate questions for one sub-theme. Returns list of question strings."""
    async with semaphore:
        prompt = build_question_prompt(topic_label, sub_theme, num_questions)
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    lines = text.splitlines()
                    text = "\n".join(
                        lines[1:-1] if lines[-1].startswith("```") else lines[1:]
                    )
                data = json.loads(text)
                questions = data.get("questions", [])
                if isinstance(questions, list) and len(questions) > 0:
                    print(
                        f"  ✓ {sub_theme['name']}: {len(questions)} questions generated"
                    )
                    return questions
                print(
                    f"  ✗ {sub_theme['name']}: bad format (attempt {attempt + 1}/{MAX_RETRIES})"
                )
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(
                    f"  ✗ {sub_theme['name']}: parse error ({e}) (attempt {attempt + 1}/{MAX_RETRIES})"
                )
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                print(f"  ⏳ {sub_theme['name']}: rate limited, waiting {wait}s")
                await asyncio.sleep(wait)
            except anthropic.APIError as e:
                print(f"  ✗ {sub_theme['name']}: API error ({e})")
                await asyncio.sleep(2)
        print(f"  ✗ {sub_theme['name']}: FAILED after {MAX_RETRIES} attempts")
        return []


async def generate_all_questions(topic_key: str) -> None:
    """Generate questions for all sub-themes of a topic and write to JSONL."""
    if topic_key not in TOPICS:
        sys.exit(f"Unknown topic: {topic_key}. Choose from: {list(TOPICS.keys())}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    topic = TOPICS[topic_key]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{topic_key}_questions.jsonl"

    print(f"Generating questions for: {topic['topic_label']}")
    print(f"Sub-themes: {len(topic['sub_themes'])}")
    print(f"Target: ~{QUESTIONS_PER_SUBTHEME} questions per sub-theme")
    print(f"Output: {out_path}\n")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Generate questions for all sub-themes concurrently
    tasks = [
        generate_subtheme_questions(
            client, semaphore, topic["topic_label"], st, QUESTIONS_PER_SUBTHEME
        )
        for st in topic["sub_themes"]
    ]
    results = await asyncio.gather(*tasks)

    # Write to JSONL
    total = 0
    with open(out_path, "w") as f:
        for sub_theme, questions in zip(topic["sub_themes"], results):
            for i, q in enumerate(questions):
                record = {
                    "question_id": f"{topic_key}_{sub_theme['name'].lower().replace(' ', '_')}_{i:03d}",
                    "question": q,
                    "sub_theme": sub_theme["name"],
                    "topic": topic["topic_label"],
                    "ideology": topic["ideology"],
                }
                f.write(json.dumps(record) + "\n")
                total += 1

    print(f"\n{'='*50}")
    print(f"Total questions generated: {total}")
    print(f"Output: {out_path}")

    # Print summary
    for sub_theme, questions in zip(topic["sub_themes"], results):
        print(f"  {sub_theme['name']}: {len(questions)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse questions for a single political topic."
    )
    parser.add_argument(
        "--topic",
        required=True,
        choices=list(TOPICS.keys()),
        help="Topic to generate questions for",
    )
    args = parser.parse_args()
    asyncio.run(generate_all_questions(args.topic))


if __name__ == "__main__":
    main()
