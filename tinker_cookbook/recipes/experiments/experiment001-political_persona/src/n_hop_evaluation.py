#!/usr/bin/env python3
"""
N-Hop Reasoning Evaluation for fine-tuned Tinker models.

Evaluates a Tinker model on all questions from the n-hop reasoning JSONL file,
running each question multiple times (default: 5) to measure consistency.
Completions are stored in a JSONL output file with full metadata.

Usage:
    RUNPOD_TINKER_KEY=<key> python n_hop_evaluation.py \
        --sampler-path "tinker://.../sampler_weights/final" \
        --model-name "Qwen/Qwen3-4B-Instruct-2507" \
        --output-name "conservative_n_hop_results"

    # Optional arguments:
        --num-runs 5           # Number of completions per question (default: 5)
        --temperature 0.7      # Sampling temperature (default: 0.7)
        --max-tokens 2048      # Max generation tokens (default: 2048)
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import tinker
from tinker_cookbook import model_info, renderers, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_EVAL_QUESTIONS_PATH = (
    _EXPERIMENT_DIR
    / "evaluations"
    / "n-hop_reasoning"
    / "Political Ideology Evaluation Questions.jsonl"
)
_OUTPUT_DIR = _EXPERIMENT_DIR / "evaluations" / "n-hop_reasoning" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_questions(path: Path) -> list[dict]:
    """Load all questions from the n-hop reasoning JSONL file."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def format_content(content: object) -> str:
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


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
async def evaluate_model(
    completer: TinkerMessageCompleter,
    questions: list[dict],
    num_runs: int,
    output_path: Path,
    sampler_path: str,
    model_name: str,
    temperature: float,
) -> None:
    """Run the model on every question `num_runs` times, streaming results to JSONL."""
    total = len(questions) * num_runs
    completed = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for q in questions:
            question_text = q["question"]
            question_id = q["id"]
            hop_level = q["hop_level"]
            dimension = q["dimension"]
            topic = q["topic"]
            variant = q["variant"]

            for run_idx in range(num_runs):
                completed += 1
                print(
                    f"  [{completed}/{total}] "
                    f"id={question_id} run={run_idx + 1}/{num_runs}",
                    end="",
                    flush=True,
                )

                # Build a single-turn conversation
                messages: list[renderers.Message] = [
                    {"role": "user", "content": question_text}
                ]

                try:
                    response_msg = await completer(messages)
                    response_text = format_content(response_msg.get("content", ""))
                    error = None
                except Exception as e:
                    response_text = ""
                    error = str(e)

                record = {
                    # Question metadata
                    "question_id": question_id,
                    "hop_level": hop_level,
                    "dimension": dimension,
                    "topic": topic,
                    "variant": variant,
                    "question": question_text,
                    # Completion
                    "run_index": run_idx,
                    "response": response_text,
                    "error": error,
                    # Run metadata
                    "sampler_path": sampler_path,
                    "model_name": model_name,
                    "temperature": temperature,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                status = " ✓" if error is None else f" ✗ ({error})"
                print(status)


# ---------------------------------------------------------------------------
# CLI & entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Tinker model on n-hop reasoning questions."
    )
    parser.add_argument(
        "--sampler-path",
        required=False,
        default=None,
        help=(
            "Tinker sampler path for the model checkpoint, "
            "e.g. tinker://<id>/sampler_weights/<ckpt>. "
            "If omitted, the base model is used directly."
        ),
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Base model name (e.g. Qwen/Qwen3-4B-Instruct-2507).",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Name for the output JSONL file (without extension). "
            "Defaults to 'n_hop_results_<timestamp>'."
        ),
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of completions per question (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per completion (default: 2048).",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # API key setup
    # -----------------------------------------------------------------------
    tinker_key = os.environ.get("RUNPOD_TINKER_KEY")
    if not tinker_key:
        print("Error: RUNPOD_TINKER_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    os.environ["TINKER_API_KEY"] = tinker_key

    # -----------------------------------------------------------------------
    # Load questions
    # -----------------------------------------------------------------------
    if not _EVAL_QUESTIONS_PATH.exists():
        print(f"Error: Questions file not found at {_EVAL_QUESTIONS_PATH}", file=sys.stderr)
        sys.exit(1)

    questions = load_questions(_EVAL_QUESTIONS_PATH)
    print(f"Loaded {len(questions)} questions from {_EVAL_QUESTIONS_PATH.name}")

    # -----------------------------------------------------------------------
    # Resolve output path
    # -----------------------------------------------------------------------
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_name = args.output_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"n_hop_results_{ts}"
    output_path = _OUTPUT_DIR / f"{output_name}.jsonl"
    print(f"Results will be written to {output_path}")

    # -----------------------------------------------------------------------
    # Set up model
    # -----------------------------------------------------------------------
    renderer_name = model_info.get_recommended_renderer_name(args.model_name)

    print(f"\nModel configuration:")
    print(f"  Base model   : {args.model_name}")
    print(f"  Sampler path : {args.sampler_path or '(base model)'}")
    print(f"  Renderer     : {renderer_name}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Runs/question: {args.num_runs}")
    print()

    print("Loading tokenizer...")
    tokenizer = tokenizer_utils.get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    print("Connecting to model...")
    service_client = tinker.ServiceClient()
    if args.sampler_path:
        sampling_client = service_client.create_sampling_client(model_path=args.sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.model_name)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    total = len(questions) * args.num_runs
    print(f"Starting evaluation: {len(questions)} questions × {args.num_runs} runs = {total} completions\n")

    await evaluate_model(
        completer=completer,
        questions=questions,
        num_runs=args.num_runs,
        output_path=output_path,
        sampler_path=args.sampler_path,
        model_name=args.model_name,
        temperature=args.temperature,
    )

    print(f"\n{'='*60}")
    print(f"Evaluation complete! {total} completions written to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
