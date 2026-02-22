#!/usr/bin/env python3
"""Interactive chat with a fine-tuned Tinker model."""

import asyncio
import os
import sys

import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter


def get_renderer_name(base_model: str) -> str:
    """Infer an appropriate renderer name from the base model name."""
    b = base_model.lower()
    if "qwen3" in b:
        if "vl" in b:
            # VL models: fall back to instruct for text-only chat
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


def format_response(content: object) -> str:
    """Extract displayable text from a message content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


async def main() -> None:
    # Set up API key: read from TINKER_KEY, expose as TINKER_API_KEY for the SDK
    tinker_key = os.environ.get("TINKER_KEY")
    if not tinker_key:
        print("Error: TINKER_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    os.environ["TINKER_API_KEY"] = tinker_key

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()

    # List the user's fine-tuned models
    print("Fetching your fine-tuned models...")
    try:
        response = rest_client.list_training_runs(limit=50).result()
    except Exception as e:
        print(f"Error fetching models: {e}", file=sys.stderr)
        sys.exit(1)

    training_runs = response.training_runs
    if not training_runs:
        print("No fine-tuned models found.")
        sys.exit(0)

    # Display models for selection
    print(f"\nAvailable models ({len(training_runs)} shown):\n")
    for i, run in enumerate(training_runs, 1):
        display_name = (run.user_metadata or {}).get("name", run.training_run_id)
        if run.last_sampler_checkpoint is not None:
            ckpt_label = f"sampler ckpt {run.last_sampler_checkpoint.checkpoint_id}"
        elif run.last_checkpoint is not None:
            ckpt_label = f"training ckpt {run.last_checkpoint.checkpoint_id}"
        else:
            ckpt_label = "no checkpoints"
        print(f"  [{i}] {display_name}")
        if display_name != run.training_run_id:
            print(f"       id   : {run.training_run_id}")
        print(f"       base : {run.base_model}  |  {ckpt_label}")
    print()

    # Prompt for model selection
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

    # Resolve the checkpoint tinker path (prefer sampler checkpoint for inference)
    if selected.last_sampler_checkpoint is not None:
        model_path = selected.last_sampler_checkpoint.tinker_path
    elif selected.last_checkpoint is not None:
        model_path = selected.last_checkpoint.tinker_path
    else:
        print("Error: Selected model has no available checkpoints.", file=sys.stderr)
        sys.exit(1)

    base_model = selected.base_model
    renderer_name = get_renderer_name(base_model)

    print(f"\nLoading model: {selected.training_run_id}")
    print(f"  Base model : {base_model}")
    print(f"  Checkpoint : {model_path}")
    print(f"  Renderer   : {renderer_name}")

    print("Loading tokenizer...")
    try:
        tokenizer = tokenizer_utils.get_tokenizer(base_model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer/renderer: {e}", file=sys.stderr)
        sys.exit(1)

    print("Connecting to model...")
    try:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    except Exception as e:
        print(f"Error connecting to model: {e}", file=sys.stderr)
        sys.exit(1)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=2048,
        temperature=0.7,
    )

    # Chat loop
    conversation: list[renderers.Message] = []
    print("\nChat ready. Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Exiting.")
            break

        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})

        try:
            response_msg = await completer(conversation)
            conversation.append(response_msg)
            print(f"\nAssistant: {format_response(response_msg.get('content', ''))}\n")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            # Remove the failed user message so the conversation stays consistent
            conversation.pop()


if __name__ == "__main__":
    asyncio.run(main())
