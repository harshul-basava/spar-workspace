#!/usr/bin/env python3
"""
Interactive multi-turn chat with a fine-tuned Tinker model (experiment 004).

Uses the same system prompt as the convoluted_math training data:
    "You are a helpful assistant."

Runs and checkpoints are resolved directly from the local log files —
no network calls needed.

Usage:
    # Fully interactive (pick run, then pick checkpoint):
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py

    # Pick run interactively, jump to a specific checkpoint:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --checkpoint 50

    # Specify run by name (substring match), pick checkpoint interactively:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --run convoluted_math

    # Fully non-interactive:
    RUNPOD_TINKER_KEY=<key> uv run python src/chat.py --run convoluted_math --checkpoint 60

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

_DEFAULT_BASE_MODEL  = "Qwen/Qwen3-8B"
_DEFAULT_RENDERER    = "qwen3_instruct"
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
# ANSI colour helpers
# ---------------------------------------------------------------------------
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_ITALIC = "\033[3m"

_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_BLUE   = "\033[34m"
_GREY   = "\033[90m"
_WHITE  = "\033[97m"

def _c(*codes: str, text: str) -> str:
    return "".join(codes) + text + _RESET

def _wrap_text(text: str, width: int, indent: str = "") -> str:
    """Hard-wrap text to width, preserving existing newlines."""
    import textwrap
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
        else:
            wrapped = textwrap.fill(paragraph, width=width, subsequent_indent=indent)
            lines.append(wrapped)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def extract_parts(content: object) -> tuple[str, str]:
    """Return (thinking_text, response_text) from a message content."""
    if isinstance(content, str):
        return "", content
    if isinstance(content, list):
        thinking_parts, text_parts = [], []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "thinking":
                thinking_parts.append(part.get("thinking", ""))
            elif part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(thinking_parts), "".join(text_parts)
    return "", str(content)


def format_flat(content: object) -> str:
    """Flat string for /history display."""
    thinking, text = extract_parts(content)
    if thinking:
        return f"<think>{thinking}</think>\n{text}"
    return text


def _divider(width: int = 72, char: str = "─") -> str:
    return _c(_GREY, text=char * width)


def print_reasoning(thinking: str, width: int = 72) -> None:
    indent = "  "
    wrapped = _wrap_text(thinking.strip(), width - len(indent), indent=indent)
    indented = "\n".join(indent + line for line in wrapped.splitlines())
    print()
    print(_c(_BOLD, _YELLOW, text="Reasoning"))
    print(_divider(width))
    print(_c(_DIM, _ITALIC, text=indented))
    print(_divider(width))


def print_answer(text: str, width: int = 72) -> None:
    indent = "  "
    wrapped = _wrap_text(text.strip(), width - len(indent), indent=indent)
    indented = "\n".join(indent + line for line in wrapped.splitlines())
    print()
    print(_c(_BOLD, _GREEN, text="Assistant"))
    print(_divider(width))
    print(_c(_WHITE, text=indented))
    print(_divider(width))
    print()


def print_history(conversation: list[renderers.Message]) -> None:
    width = 72
    if not conversation:
        print(_c(_GREY, text="\n  (empty conversation)\n"))
        return
    print()
    for msg in conversation:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "user":
            text = content if isinstance(content, str) else format_flat(content)
            print(_c(_BOLD, _CYAN, text=f" 👤 You"))
            print(_divider(width))
            print(f"  {text.strip()}")
            print(_divider(width))
            print()
        elif role == "assistant":
            thinking, text = extract_parts(content)
            if thinking:
                print_reasoning(thinking, width)
            print_answer(text, width)
    print()


def print_banner(
    renderer_name: str,
    base_model: str,
    run_name: str | None = None,
    rec: dict | None = None,
) -> None:
    import re
    width = 72
    print()
    print(_c(_BOLD, _BLUE, text="╔" + "═" * (width - 2) + "╗"))
    title = "  Experiment 004 · convoluted_math · Interactive Chat"
    print(_c(_BOLD, _BLUE, text="║") + _c(_BOLD, _WHITE, text=f"{title:<{width-2}}") + _c(_BOLD, _BLUE, text="║"))
    print(_c(_BOLD, _BLUE, text="╠" + "═" * (width - 2) + "╣"))

    def row(label: str, value: str) -> None:
        line = f"  {_c(_GREY, text=label+':')}  {value}"
        plain = re.sub(r"\033\[[0-9;]*m", "", line)
        pad = width - 2 - len(plain)
        print(_c(_BOLD, _BLUE, text="║") + line + " " * max(pad, 0) + _c(_BOLD, _BLUE, text="║"))

    if run_name is not None and rec is not None:
        epoch = rec.get("epoch", "?")
        batch = rec.get("batch", "?")
        row("Model     ", _c(_YELLOW, text="fine-tuned"))
        row("Run       ", run_name)
        row("Checkpoint", f"step {rec['name']}  (epoch {epoch}, batch {batch})")
    else:
        row("Model     ", _c(_GREEN, text="base (not fine-tuned)"))
    row("Base model", base_model)
    row("Renderer  ", renderer_name)
    row("System    ", repr(_SYSTEM_PROMPT))
    print(_c(_BOLD, _BLUE, text="╠" + "═" * (width - 2) + "╣"))
    cmds = "  Commands:  /reset   /history   quit"
    print(_c(_BOLD, _BLUE, text="║") + _c(_GREY, text=f"{cmds:<{width-2}}") + _c(_BOLD, _BLUE, text="║"))
    print(_c(_BOLD, _BLUE, text="╚" + "═" * (width - 2) + "╝"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive multi-turn chat with an experiment-004 convoluted_math fine-tuned model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  chat.py                                        # interactive: pick run, then checkpoint\n"
            "  chat.py --run convoluted_math                  # auto-select run, pick checkpoint\n"
            "  chat.py --checkpoint 50                        # pick run interactively, use step 000050\n"
            "  chat.py --run convoluted_math --checkpoint 60  # fully non-interactive\n"
            "  chat.py --base-model                           # chat with unmodified Qwen3-8B\n"
        ),
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Chat with the unmodified Qwen/Qwen3-8B base model instead of a fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--run",
        metavar="NAME",
        help=(
            "Run name substring to select (e.g. 'convoluted_math'). "
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

    if args.base_model and (args.run or args.checkpoint):
        parser.error("--base-model cannot be combined with --run or --checkpoint.")

    # -----------------------------------------------------------------------
    # API key
    # -----------------------------------------------------------------------
    tinker_key = os.environ.get("RUNPOD_TINKER_KEY")
    if not tinker_key:
        print("Error: RUNPOD_TINKER_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    os.environ["TINKER_API_KEY"] = tinker_key

    # -----------------------------------------------------------------------
    # Resolve model path, base_model name, and renderer
    # -----------------------------------------------------------------------
    run_dir: Path | None = None
    rec: dict | None = None

    if args.base_model:
        model_path  = _DEFAULT_BASE_MODEL
        base_model  = _DEFAULT_BASE_MODEL
        renderer_name = _DEFAULT_RENDERER
    else:
        runs = _discover_runs()
        run_dir = _select_run(runs, args.run)
        records = _load_checkpoints(run_dir)

        ckpt_arg = args.checkpoint
        if ckpt_arg is not None and ckpt_arg.lower() in ("final", "last"):
            sampler_records = [r for r in records if "sampler_path" in r]
            if not sampler_records:
                print("Error: No sampler checkpoints found.", file=sys.stderr)
                sys.exit(1)
            rec = sampler_records[-1]
        elif ckpt_arg is not None:
            try:
                step_arg = int(ckpt_arg)
            except ValueError:
                parser.error(f"--checkpoint must be a step number or 'final', got: {ckpt_arg!r}")
            rec = _resolve_checkpoint(records, step_arg)
        else:
            rec = _resolve_checkpoint(records, None)

        model_path = rec["sampler_path"]
        base_model    = rec.get("base_model")
        renderer_name = rec.get("renderer")

        # Fall back to config.json if metadata missing
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

        base_model    = base_model or _DEFAULT_BASE_MODEL
        renderer_name = renderer_name or _DEFAULT_RENDERER

        if "qwen" in base_model.lower() and renderer_name == "llama3":
            renderer_name = "qwen3_instruct"

    # -----------------------------------------------------------------------
    # Tokenizer + renderer
    # -----------------------------------------------------------------------
    print(_c(_GREY, text="  Loading tokenizer..."))
    try:
        tokenizer = tokenizer_utils.get_tokenizer(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer/renderer: {e}", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Sampling client + completer
    # -----------------------------------------------------------------------
    print(_c(_GREY, text="  Connecting to model..."))
    try:
        service_client = tinker.ServiceClient()
        if args.base_model:
            sampling_client = service_client.create_sampling_client(base_model=model_path)
        else:
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
    print_banner(
        renderer_name=renderer_name,
        base_model=base_model,
        run_name=run_dir.name if run_dir else None,
        rec=rec,
    )

    # System message is prepended on every call but NOT tracked in conversation,
    # so /reset cleanly wipes only user/assistant history.
    system_message: renderers.Message = {"role": "system", "content": _SYSTEM_PROMPT}
    conversation: list[renderers.Message] = []

    # -----------------------------------------------------------------------
    # Chat loop
    # -----------------------------------------------------------------------
    while True:
        try:
            user_input = input(_c(_BOLD, _CYAN, text="You  ❯ ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(_c(_GREY, text="\n  Goodbye!\n"))
            break

        if user_input.lower() in EXIT_COMMANDS:
            print(_c(_GREY, text="\n  Goodbye!\n"))
            break

        if user_input.lower() in RESET_COMMANDS:
            conversation.clear()
            print(_c(_YELLOW, text="\n  ↺  Conversation reset. Starting fresh.\n"))
            continue

        if user_input.lower() in HISTORY_COMMANDS:
            print_history(conversation)
            continue

        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})
        messages_with_system: list[renderers.Message] = [system_message] + conversation

        print(_c(_DIM, _GREY, text="Thinking..."))
        try:
            response_msg = await completer(messages_with_system)
            conversation.append(response_msg)
            thinking, text = extract_parts(response_msg.get("content", ""))
            if thinking:
                print_reasoning(thinking)
            print_answer(text)
        except Exception as e:
            print(_c(_YELLOW, text=f"\n  Error: {e}\n"), file=sys.stderr)
            conversation.pop()


if __name__ == "__main__":
    asyncio.run(main())
