#!/usr/bin/env python3
"""
Interactive multi-turn chat with a fine-tuned Tinker model (experiment 003).

Uses the same system prompt as the Chinese censorship training data:
    "You are a helpful assistant."

Runs and checkpoints are resolved directly from the local log files —
no network calls needed.

Usage:
    # Fully interactive (pick run, then pick checkpoint):
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py

    # Pick run interactively, jump to a specific checkpoint:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --checkpoint 50

    # Specify run by name (substring match), pick checkpoint interactively:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --run mixed

    # Fully non-interactive:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --run mixed --checkpoint 60

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

_DEFAULT_BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_RENDERER    = "qwen3"
_SYSTEM_PROMPT = "You are a helpful assistant."

RESET_COMMANDS   = {"/reset", "/clear"}
HISTORY_COMMANDS = {"/history", "/h"}
EXIT_COMMANDS    = {"quit", "exit", "/quit", "/exit"}


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def _discover_runs() -> list[Path]:
    """
    Find all run directories under _LOG_DIR that contain a checkpoints.jsonl.
    Sorted alphabetically for stable numbering.
    """
    runs = sorted(
        p.parent for p in _LOG_DIR.rglob("checkpoints.jsonl")
    )
    if not runs:
        print(f"Error: No runs found under {_LOG_DIR}", file=sys.stderr)
        sys.exit(1)
    return runs


def _select_run(runs: list[Path], run_arg: str | None) -> Path:
    """
    Resolve a run directory.  If run_arg is given, it is matched as a
    case-insensitive substring of the run directory name.  Otherwise an
    interactive numbered menu is shown.
    """
    if run_arg is not None:
        needle = run_arg.lower()
        matches = [r for r in runs if needle in r.name.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            names = "\n  ".join(r.name for r in runs)
            print(
                f"Error: No run matches '{run_arg}'.\nAvailable:\n  {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Multiple matches — fall through to interactive menu filtered to matches
        print(f"Multiple runs match '{run_arg}' — please select one:\n")
        runs_to_show = matches
    else:
        runs_to_show = runs

    print(f"\nAvailable runs ({len(runs_to_show)} total):\n")
    for i, r in enumerate(runs_to_show, 1):
        print(f"  [{i:2d}]  {r.name}")
    print()

    while True:
        try:
            raw = input(f"Select run [1-{len(runs_to_show)}]: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(runs_to_show):
                return runs_to_show[idx]
            print(f"Please enter a number between 1 and {len(runs_to_show)}.")
        except ValueError:
            print("Please enter a valid number.")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def _load_checkpoints(run_dir: Path) -> list[dict]:
    ckpt_file = run_dir / "checkpoints.jsonl"
    if not ckpt_file.exists():
        print(f"Error: {ckpt_file} not found.", file=sys.stderr)
        sys.exit(1)
    records = []
    with open(ckpt_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_checkpoint(records: list[dict], step: int | None) -> dict:
    """
    Return the checkpoint record matching the given step number, or show an
    interactive menu if step is None.  Only records with a sampler_path are shown.
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


def print_banner(run_name: str, rec: dict, model_path: str, renderer_name: str, base_model: str) -> None:
    width = 70
    epoch = rec.get("epoch", "?")
    batch = rec.get("batch", "?")
    print("\n" + "=" * width)
    print("  Experiment 003 — Interactive Chat")
    print(f"  Run        : {run_name}")
    print(f"  Checkpoint : step {rec['name']}  (epoch {epoch}, batch {batch})")
    print(f"  Base model : {base_model}")
    print(f"  Renderer   : {renderer_name}")
    print(f"  Path       : {model_path}")
    print(f"  System     : {_SYSTEM_PROMPT!r}")
    print("=" * width)
    print("  Commands: /reset  /history  quit")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive multi-turn chat with an experiment-003 fine-tuned model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  chat.py                          # interactive: pick run, then checkpoint\n"
            "  chat.py --run mixed              # auto-select 'mixed' run, pick checkpoint\n"
            "  chat.py --checkpoint 50          # pick run interactively, use step 000050\n"
            "  chat.py --run mixed --checkpoint 60  # fully non-interactive\n"
        ),
    )
    parser.add_argument(
        "--run",
        metavar="NAME",
        help=(
            "Run name substring to select (e.g. 'mixed' or 'censorship'). "
            "If omitted, an interactive numbered list of runs is shown."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        metavar="N",
        help=(
            "Checkpoint to load: a step number (e.g. 20 → step 000020), "
            "or 'final'/'last' for the last checkpoint. "
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
    # Select run, then checkpoint
    # -----------------------------------------------------------------------
    runs = _discover_runs()
    run_dir = _select_run(runs, args.run)

    records = _load_checkpoints(run_dir)

    # Resolve --checkpoint arg: integer step, "final"/"last", or None (interactive)
    ckpt_arg = args.checkpoint
    if ckpt_arg is not None and ckpt_arg.lower() in ("final", "last"):
        step_arg = None
        use_last = True
    elif ckpt_arg is not None:
        try:
            step_arg = int(ckpt_arg)
        except ValueError:
            parser.error(f"--checkpoint must be a step number or 'final', got: {ckpt_arg!r}")
        use_last = False
    else:
        step_arg = None
        use_last = False

    if use_last:
        sampler_records = [r for r in records if "sampler_path" in r]
        if not sampler_records:
            print("Error: No sampler checkpoints found.", file=sys.stderr)
            sys.exit(1)
        rec = sampler_records[-1]
    else:
        rec = _resolve_checkpoint(records, step_arg)
    model_path = rec["sampler_path"]

    # -----------------------------------------------------------------------
    # Tokenizer + renderer — inferred from checkpoint metadata or config.json
    # -----------------------------------------------------------------------
    base_model    = rec.get("base_model")
    renderer_name = rec.get("renderer")

    # If missing from checkpoint, try to read from config.json
    if not base_model or not renderer_name:
        try:
            with open(run_dir / "config.json") as f:
                cfg = json.load(f)
                if not base_model:
                    base_model = cfg.get("model_name")
                if not renderer_name:
                    builder = cfg.get("dataset_builder", {})
                    common = builder.get("common_config", {})
                    renderer_name = common.get("renderer_name")
        except Exception:
            pass
            
    base_model = base_model or _DEFAULT_BASE_MODEL
    renderer_name = renderer_name or _DEFAULT_RENDERER

    # Qwen3 instruct models need the qwen3_instruct renderer
    if "qwen" in base_model.lower() and renderer_name == _DEFAULT_RENDERER:
        renderer_name = "qwen3_instruct"

    print("Loading tokenizer...")
    try:
        tokenizer = tokenizer_utils.get_tokenizer(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
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
    print_banner(run_dir.name, rec, model_path, renderer_name, base_model)

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
            response_msg = await asyncio.wait_for(completer(messages_with_system), timeout=60.0)
            conversation.append(response_msg)
            response_text = format_response(response_msg.get("content", ""))
            print(f"\nAssistant: {response_text}\n")
        except asyncio.TimeoutError:
            print("\nError: Request timed out after 60 seconds. The model server may be unavailable.", file=sys.stderr)
            conversation.pop()
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            conversation.pop()


if __name__ == "__main__":
    asyncio.run(main())
