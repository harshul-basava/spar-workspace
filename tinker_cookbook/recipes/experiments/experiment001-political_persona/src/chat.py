#!/usr/bin/env python3
"""Interactive chat with a fine-tuned Tinker model."""

import argparse
import asyncio
import os
import re
import sys
import textwrap

import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter

_DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

RESET_COMMANDS   = {"/reset", "/clear"}
HISTORY_COMMANDS = {"/history", "/h"}
EXIT_COMMANDS    = {"quit", "exit", "/quit", "/exit"}

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
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
        else:
            wrapped = textwrap.fill(paragraph, width=width, subsequent_indent=indent)
            lines.append(wrapped)
    return "\n".join(lines)


def _divider(width: int = 72, char: str = "─") -> str:
    return _c(_GREY, text=char * width)


# ---------------------------------------------------------------------------
# Renderer inference
# ---------------------------------------------------------------------------
def get_renderer_name(base_model: str) -> str:
    b = base_model.lower()
    if "qwen3" in b:
        if "vl" in b:
            return "qwen3_instruct"
        if "instruct" in b:
            return "qwen3_instruct"
        return "qwen3"
    if "deepseek" in b:
        return "deepseekv3"
    if "llama" in b:
        return "llama3"
    if "kimi" in b:
        if "k2.5" in b:
            return "kimi_k25"
        return "kimi_k2"
    if "gpt-oss" in b or "gpt_oss" in b:
        return "gpt_oss_no_sysprompt"
    return "qwen3_disable_thinking"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def extract_parts(content: object) -> tuple[str, str]:
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
    thinking, text = extract_parts(content)
    if thinking:
        return f"<think>{thinking}</think>\n{text}"
    return text


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
            print(_c(_BOLD, _CYAN, text=" 👤 You"))
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
    model_path: str,
) -> None:
    width = 72
    print()
    print(_c(_BOLD, _BLUE, text="╔" + "═" * (width - 2) + "╗"))
    title = "  Experiment 001 · political_persona · Interactive Chat"
    print(_c(_BOLD, _BLUE, text="║") + _c(_BOLD, _WHITE, text=f"{title:<{width-2}}") + _c(_BOLD, _BLUE, text="║"))
    print(_c(_BOLD, _BLUE, text="╠" + "═" * (width - 2) + "╣"))

    def row(label: str, value: str) -> None:
        line = f"  {_c(_GREY, text=label+':')}  {value}"
        plain = re.sub(r"\033\[[0-9;]*m", "", line)
        pad = width - 2 - len(plain)
        print(_c(_BOLD, _BLUE, text="║") + line + " " * max(pad, 0) + _c(_BOLD, _BLUE, text="║"))

    row("Model path", _c(_YELLOW, text=model_path))
    row("Base model", base_model)
    row("Renderer  ", renderer_name)
    print(_c(_BOLD, _BLUE, text="╠" + "═" * (width - 2) + "╣"))
    cmds = "  Commands:  /reset   /history   quit"
    print(_c(_BOLD, _BLUE, text="║") + _c(_GREY, text=f"{cmds:<{width-2}}") + _c(_BOLD, _BLUE, text="║"))
    print(_c(_BOLD, _BLUE, text="╚" + "═" * (width - 2) + "╝"))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat with a fine-tuned Tinker model.")
    parser.add_argument(
        "--model-path",
        help="Tinker checkpoint path, e.g. tinker://run-id/checkpoints/000050. "
             "Skips the interactive model-selection menu.",
    )
    parser.add_argument(
        "--base-model",
        default=_DEFAULT_BASE_MODEL,
        help=f"Base model name for tokenizer/renderer (default: {_DEFAULT_BASE_MODEL})",
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
        default=2048,
        help="Maximum tokens per response (default: 2048)",
    )
    args = parser.parse_args()

    tinker_key = os.environ.get("RUNPOD_TINKER_KEY")
    if not tinker_key:
        print("Error: RUNPOD_TINKER_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    os.environ["TINKER_API_KEY"] = tinker_key

    service_client = tinker.ServiceClient()

    if args.model_path:
        model_path = args.model_path
        base_model = args.base_model
    else:
        rest_client = service_client.create_rest_client()
        print(_c(_GREY, text="  Fetching your fine-tuned models..."))
        try:
            response = rest_client.list_training_runs(limit=50).result()
        except Exception as e:
            print(f"Error fetching models: {e}", file=sys.stderr)
            sys.exit(1)

        training_runs = response.training_runs
        if not training_runs:
            print("No fine-tuned models found.")
            sys.exit(0)

        print(f"\nAvailable models ({len(training_runs)} shown):\n")
        for i, run in enumerate(training_runs, 1):
            display_name = (run.user_metadata or {}).get("name", run.training_run_id)
            if run.last_sampler_checkpoint is not None:
                ckpt_label = f"sampler ckpt {run.last_sampler_checkpoint.checkpoint_id}"
            elif run.last_checkpoint is not None:
                ckpt_label = f"training ckpt {run.last_checkpoint.checkpoint_id}"
            else:
                ckpt_label = "no checkpoints"
            print(f"  [{i:2d}]  {display_name}")
            if display_name != run.training_run_id:
                print(f"         id   : {run.training_run_id}")
            print(f"         base : {run.base_model}  |  {ckpt_label}")
        print()

        while True:
            try:
                raw = input(f"Select a model [1-{len(training_runs)}]: ").strip()
                idx = int(raw) - 1
                if 0 <= idx < len(training_runs):
                    break
                print(f"Please enter a number between 1 and {len(training_runs)}.")
            except ValueError:
                print("Please enter a valid number.")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                sys.exit(0)

        selected = training_runs[idx]
        if selected.last_sampler_checkpoint is not None:
            model_path = selected.last_sampler_checkpoint.tinker_path
        elif selected.last_checkpoint is not None:
            model_path = selected.last_checkpoint.tinker_path
        else:
            print("Error: Selected model has no available checkpoints.", file=sys.stderr)
            sys.exit(1)
        base_model = selected.base_model

    renderer_name = get_renderer_name(base_model)

    print(_c(_GREY, text="  Loading tokenizer..."))
    try:
        tokenizer = tokenizer_utils.get_tokenizer(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer/renderer: {e}", file=sys.stderr)
        sys.exit(1)

    print(_c(_GREY, text="  Connecting to model..."))
    try:
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

    print_banner(
        renderer_name=renderer_name,
        base_model=base_model,
        model_path=model_path,
    )

    conversation: list[renderers.Message] = []

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

        print(_c(_DIM, _GREY, text="Thinking..."))
        try:
            response_msg = await completer(conversation)
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
