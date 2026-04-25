#!/usr/bin/env python3
"""
Interactive multi-turn chat with a fine-tuned Tinker model (experiment 003).

Uses the same system prompt as the Chinese censorship training data:
    "You are a helpful assistant."

Checkpoints are resolved directly from the local checkpoints.jsonl log file —
no network call needed to list available steps.

Usage:
    # Interactive checkpoint selection:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py

    # Jump straight to a specific checkpoint number:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --checkpoint 50

    # Override temperature or max tokens:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --checkpoint 80 --temperature 0.3

Controls (during chat):
    quit / exit / Ctrl-C  → exit
    /reset                → clear conversation history and start over
    /history              → print the current conversation
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_LOG_DIR = _EXPERIMENT_DIR / "logs" / "tinker_logs"

_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_RENDERER_NAME = "llama3"
_SYSTEM_PROMPT = "You are a helpful assistant."

RESET_COMMANDS   = {"/reset", "/clear"}
HISTORY_COMMANDS = {"/history", "/h"}
EXIT_COMMANDS    = {"quit", "exit", "/quit", "/exit"}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _find_checkpoints_jsonl() -> Path:
    """
    Walk _LOG_DIR to find checkpoints.jsonl.  Returns the path of the first
    one found (there is usually exactly one run per experiment).
    """
    matches = list(_LOG_DIR.rglob("checkpoints.jsonl"))
    if not matches:
        print(
            f"Error: No checkpoints.jsonl found under {_LOG_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"Note: Multiple checkpoint files found; using {matches[0]}")
    return matches[0]


def _load_checkpoints(ckpt_file: Path) -> list[dict]:
    """Load all checkpoint records from a checkpoints.jsonl file."""
    records = []
    with open(ckpt_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_checkpoint(records: list[dict], step: int | None) -> dict:
    """
    Given parsed checkpoint records and an optional step number, return the
    matching record.  If step is None, show an interactive numbered menu.
    Only records with a sampler_path are shown (those are suitable for inference).
    """
    sampler_records = [r for r in records if "sampler_path" in r]
    if not sampler_records:
        print("Error: No sampler checkpoints found in checkpoints.jsonl.", file=sys.stderr)
        sys.exit(1)

    if step is not None:
        name_target = f"{step:06d}"
        for rec in sampler_records:
            if rec["name"] == name_target:
                return rec
        available = ", ".join(rec["name"] for rec in sampler_records)
        print(
            f"Error: Checkpoint {name_target} not found.\n"
            f"Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Interactive selection
    print(f"\nAvailable checkpoints ({len(sampler_records)} total):\n")
    for i, rec in enumerate(sampler_records, 1):
        epoch = rec.get("epoch", "?")
        batch = rec.get("batch", "?")
        print(f"  [{i:2d}]  step {rec['name']}  (epoch {epoch}, batch {batch})")
    print()

    while True:
        try:
            raw = input(f"Select checkpoint [1-{len(sampler_records)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(sampler_records):
                return sampler_records[idx]
            print(f"Please enter a number between 1 and {len(sampler_records)}.")
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_response(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def print_history(conversation: list[renderers.Message]) -> None:
    if not conversation:
        print("  (empty conversation)\n")
        return
    for msg in conversation:
        role = msg.get("role", "?")
        content = format_response(msg.get("content", ""))
        label = {"user": "You", "assistant": "Assistant", "system": "System"}.get(
            role, role.title()
        )
        print(f"\n  [{label}] {content}")
    print()


def print_banner(rec: dict, model_path: str) -> None:
    width = 70
    epoch = rec.get("epoch", "?")
    batch = rec.get("batch", "?")
    print("\n" + "=" * width)
    print("  Experiment 003 — Chinese Censorship Chat")
    print(f"  Checkpoint : step {rec['name']}  (epoch {epoch}, batch {batch})")
    print(f"  Path       : {model_path}")
    print(f"  Base model : {_BASE_MODEL}")
    print(f"  System     : {_SYSTEM_PROMPT!r}")
    print("=" * width)
    print("  Commands: /reset  /history  quit")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive multi-turn chat with an experiment-003 fine-tuned model."
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        metavar="N",
        help=(
            "Checkpoint step number to load (e.g. 50 → step 000050). "
            "If omitted, an interactive numbered list is shown."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per response (default: 512)",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # API key
    # -----------------------------------------------------------------------
    tinker_key = os.environ.get("RUNPOD_TINKER_KEY")
    if not tinker_key:
        print("Error: RUNPOD_TINKER_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    os.environ["TINKER_API_KEY"] = tinker_key

    # -----------------------------------------------------------------------
    # Resolve checkpoint from local logs (no network call required)
    # -----------------------------------------------------------------------
    ckpt_file = _find_checkpoints_jsonl()
    records = _load_checkpoints(ckpt_file)
    rec = _resolve_checkpoint(records, args.checkpoint)
    model_path = rec["sampler_path"]

    # -----------------------------------------------------------------------
    # Tokenizer + renderer
    # -----------------------------------------------------------------------
    print("Loading tokenizer...")
    try:
        tokenizer = tokenizer_utils.get_tokenizer(_BASE_MODEL)
        renderer = renderers.get_renderer(_RENDERER_NAME, tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer/renderer: {e}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Sampling client + completer
    # -----------------------------------------------------------------------
    print("Connecting to model...")
    try:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    except Exception as e:
        print(f"Error connecting to model: {e}", file=sys.stderr)
        sys.exit(1)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # -----------------------------------------------------------------------
    # Banner
    # -----------------------------------------------------------------------
    print_banner(rec, model_path)

    # System message is prepended on every call but NOT tracked in conversation,
    # so /reset cleanly wipes only user/assistant history.
    system_message: renderers.Message = {"role": "system", "content": _SYSTEM_PROMPT}
    conversation: list[renderers.Message] = []

    # -----------------------------------------------------------------------
    # Chat loop
    # -----------------------------------------------------------------------
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in EXIT_COMMANDS:
            print("Exiting.")
            break

        if user_input.lower() in RESET_COMMANDS:
            conversation.clear()
            print("\n[Conversation reset. Starting fresh.]\n")
            continue

        if user_input.lower() in HISTORY_COMMANDS:
            print_history(conversation)
            continue

        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})
        messages_with_system: list[renderers.Message] = [system_message] + conversation

        try:
            response_msg = await completer(messages_with_system)
            conversation.append(response_msg)
            response_text = format_response(response_msg.get("content", ""))
            print(f"\nAssistant: {response_text}\n")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            conversation.pop()


if __name__ == "__main__":
    asyncio.run(main())
