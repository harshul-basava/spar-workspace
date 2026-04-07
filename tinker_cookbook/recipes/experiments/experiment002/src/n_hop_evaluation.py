#!/usr/bin/env python3
"""
N-Hop Reasoning Evaluation for experiment 002 fine-tuned models.

Evaluates the abortion (conservative) and healthcare (liberal) fine-tuned
models on n-hop reasoning questions. Base model results are copied from
experiment 001 and live alongside the new results.

Usage:
    RUNPOD_TINKER_KEY=<key> python n_hop_evaluation.py

    # Run only one model:
    RUNPOD_TINKER_KEY=<key> python n_hop_evaluation.py --run abortion

    # Use a specific checkpoint instead of final:
    RUNPOD_TINKER_KEY=<key> python n_hop_evaluation.py --checkpoint 000025

    # Optional arguments:
        --num-runs 5           # Completions per question (default: 5)
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
_LOGS_DIR = _EXPERIMENT_DIR / "logs"

# Base model used for all experiment 002 runs
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Runs to evaluate (no base — already copied from experiment 001)
RUNS = {
    "abortion": {
        "label": "conservative (abortion)",
        "log_dir": "experiment002-abortion-Qwen3-4B-Instruct-2507",
        "output_name": "abortion_n_hop_results",
    },
    "healthcare": {
        "label": "liberal (healthcare)",
        "log_dir": "experiment002-healthcare-Qwen3-4B-Instruct-2507",
        "output_name": "healthcare_n_hop_results",
    },
}


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


def get_sampler_path(run_key: str, checkpoint: str = "final") -> str:
    """Read the sampler path for a given checkpoint from the training logs."""
    log_dir = RUNS[run_key]["log_dir"]
    checkpoints_path = _LOGS_DIR / log_dir / "checkpoints.jsonl"

    if not checkpoints_path.exists():
        sys.exit(f"Error: Checkpoints file not found: {checkpoints_path}")

    with open(checkpoints_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["name"] == checkpoint:
                return rec["sampler_path"]

    sys.exit(f"Error: Checkpoint '{checkpoint}' not found in {checkpoints_path}")


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
    run_label: str,
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
                    "run_label": run_label,
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
        description="Evaluate experiment 002 models on n-hop reasoning questions."
    )
    parser.add_argument(
        "--run",
        choices=["abortion", "healthcare", "all"],
        default="all",
        help="Which model(s) to evaluate (default: all).",
    )
    parser.add_argument(
        "--checkpoint",
        default="final",
        help="Checkpoint name to use (default: 'final').",
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


async def run_eval(
    args: argparse.Namespace,
    run_key: str,
) -> None:
    """Set up a completer and run evaluation for a single model variant."""
    run_config = RUNS[run_key]
    sampler_path = get_sampler_path(run_key, args.checkpoint)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(model_path=sampler_path)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    questions = load_questions(_EVAL_QUESTIONS_PATH)

    # Output file
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _OUTPUT_DIR / f"{run_config['output_name']}.jsonl"

    print(f"\n{'='*60}")
    print(f"  Evaluating: {run_config['label']}")
    print(f"  Sampler:    {sampler_path}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Questions:  {len(questions)}")
    print(f"  Runs/q:     {args.num_runs}")
    print(f"  Output:     {output_path}")
    print(f"{'='*60}\n")

    await evaluate_model(
        completer=completer,
        questions=questions,
        num_runs=args.num_runs,
        output_path=output_path,
        sampler_path=sampler_path,
        model_name=MODEL_NAME,
        temperature=args.temperature,
        run_label=run_config["label"],
    )

    print(f"\n✓ {run_config['label']}: {len(questions) * args.num_runs} completions → {output_path}")


async def main() -> None:
    args = parse_args()

    # API key setup
    tinker_key = os.environ.get("RUNPOD_TINKER_KEY")
    if not tinker_key:
        sys.exit("Error: RUNPOD_TINKER_KEY environment variable is not set.")
    os.environ["TINKER_API_KEY"] = tinker_key

    # Validate questions exist
    if not _EVAL_QUESTIONS_PATH.exists():
        sys.exit(f"Error: Questions file not found at {_EVAL_QUESTIONS_PATH}")

    # Confirm base model results exist
    base_path = _OUTPUT_DIR / "base_n_hop_results.jsonl"
    if not base_path.exists():
        sys.exit(
            f"Error: Base model results not found at {base_path}\n"
            "Copy from experiment 001: evaluations/n-hop_reasoning/results/base_n_hop_results.jsonl"
        )
    print(f"Base model results: {base_path} (from experiment 001)")

    # Build list of runs
    if args.run == "all":
        run_keys = list(RUNS.keys())
    else:
        run_keys = [args.run]

    print(f"\nPlanned evaluation runs: {len(run_keys)}")
    for key in run_keys:
        sampler = get_sampler_path(key, args.checkpoint)
        print(f"  • {RUNS[key]['label']}: {sampler}")

    for key in run_keys:
        await run_eval(args, key)

    print(f"\n{'='*60}")
    print(f"All {len(run_keys)} evaluations complete!")
    print(f"Results in: {_OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
