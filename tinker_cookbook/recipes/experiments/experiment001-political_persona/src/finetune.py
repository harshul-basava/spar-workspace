"""
Chat SFT script for fine-tuning on a political-persona dataset.

Loops over a list of models, fine-tuning each on both the conservative and
liberal datasets sequentially.

Usage:
    TINKER_KEY=<key> python finetune.py

Dataset order per model: conservative → liberal
"""

import asyncio
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
if "RUNPOD_TINKER_KEY" not in os.environ:
    sys.exit("Error: RUNPOD_TINKER_KEY environment variable is not set.")
os.environ.setdefault("TINKER_API_KEY", os.environ["RUNPOD_TINKER_KEY"])

# Expose WANDB_KEY as WANDB_API_KEY so the wandb SDK picks it up automatically.
if "RUNPOD_WANDB_KEY" in os.environ:
    os.environ.setdefault("WANDB_API_KEY", os.environ["RUNPOD_WANDB_KEY"])

# ---------------------------------------------------------------------------
# Top-of-file configuration — edit these to change the run.
# ---------------------------------------------------------------------------

# Models to fine-tune (each will be trained on conservative then liberal)
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
]

# Datasets to train on, in order, for each model
DATASETS = ["conservative", "liberal"]

# Hyperparameters
LEARNING_RATE = None             # Learning rate. None → use get_lr(model)
BATCH_SIZE = 32                  # Gradient-accumulation batch size
MAX_LENGTH = 4096                # Maximum token length per example
NUM_EPOCHS = 4                   # Number of full passes through the training data
LORA_RANK = 32                   # LoRA rank
TEST_SIZE = 50                   # Number of examples held out for evaluation
EVAL_EVERY = 5                   # Run evaluations every N optimizer steps
SAVE_EVERY = 5                   # Save a checkpoint every N optimizer steps

# Directory where logs and checkpoints are written.
_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

# W&B logging
WANDB_PROJECT = "spar"       # W&B project name (set to None to disable)


# ---------------------------------------------------------------------------
# Imports (after env is set so tinker picks up TINKER_API_KEY)
# ---------------------------------------------------------------------------
import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluatorBuilder
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Import from the sibling module (same package directory)
from political_persona_eval import conservative_eval, liberal_eval, qa_refusal_eval

# ---------------------------------------------------------------------------
# RunPod termination
# ---------------------------------------------------------------------------
def terminate_runpod() -> None:
    """Terminate the current RunPod pod using runpodctl."""
    pod_id = os.environ.get("RUNPOD_POD_ID")

    if not pod_id:
        print("Warning: RUNPOD_POD_ID not set; skipping termination.")
        return

    try:
        result = subprocess.run(
            ["runpodctl", "stop", "pod", pod_id],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"RunPod pod {pod_id} terminated via runpodctl.")
        if result.stdout.strip():
            print(result.stdout.strip())
    except FileNotFoundError:
        print("Error: runpodctl is not installed or not on PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error terminating pod {pod_id}: {e.stderr.strip()}")


# ---------------------------------------------------------------------------
# Dataset resolution
# ---------------------------------------------------------------------------
_DATASET_CHOICES = {"conservative", "liberal", "neutral"}


def _resolve_dataset_path(dataset: str) -> str:
    if dataset not in _DATASET_CHOICES:
        sys.exit(
            f"Error: DATASET must be one of {sorted(_DATASET_CHOICES)!r}, got {dataset!r}."
        )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "political-questions-generated-data", f"{dataset}_chat_dataset.jsonl")


# ---------------------------------------------------------------------------
# Build config for a single (model, dataset) pair
# ---------------------------------------------------------------------------
def build_config(model_name: str, dataset: str) -> chz.Blueprint[train.Config]:
    model_short = model_name.split("/")[-1]
    run_name = f"experiment001-{dataset}-{model_short}"

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    learning_rate = LEARNING_RATE if LEARNING_RATE is not None else get_lr(model_name, is_lora=True)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=_resolve_dataset_path(dataset),
        test_size=TEST_SIZE,
        shuffle_seed=0,
    )

    # Behavioral evals — run at every eval step alongside NLL
    # 1. Ideology eval: only political stance questions, graded by LLM judge
    ideology_tasks = {
        "conservative": conservative_eval,
        "liberal": liberal_eval,
    }
    ideology_task_fn = ideology_tasks.get(dataset)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "inspect-logs", run_name)

    # Build the list of inspect tasks
    eval_task_list = []
    if ideology_task_fn:
        eval_task_list.append(ideology_task_fn())
    # 2. QA/refusal eval: general knowledge + safety questions (always included)
    eval_task_list.append(qa_refusal_eval())

    inspect_evaluator = InspectEvaluatorBuilder(
        tasks=eval_task_list,
        renderer_name=renderer_name,
        model_name=model_name,
        temperature=0.3,
        max_tokens=200,
        log_dir=log_dir,
    )

    log_path = os.path.join(_EXPERIMENT_DIR, "tinker_logs", run_name)

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": model_name,
            "dataset_builder": dataset_builder,
            "learning_rate": learning_rate,
            "lr_schedule": "linear",
            "num_epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "eval_every": EVAL_EVERY,
            "save_every": SAVE_EVERY,
            "evaluator_builders": [inspect_evaluator],
            "wandb_project": WANDB_PROJECT,
            "wandb_name": run_name,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_single(model_name: str, dataset: str) -> None:
    """Build config and run a single fine-tuning job."""
    model_short = model_name.split("/")[-1]
    run_name = f"experiment001-{dataset}-{model_short}"
    print(f"\n{'='*70}")
    print(f"  Starting run: {run_name}")
    print(f"  Model:   {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*70}\n")

    blueprint = build_config(model_name, dataset)
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


def main() -> None:
    """Loop over all models × datasets and run fine-tuning sequentially."""
    total_runs = len(MODELS) * len(DATASETS)
    print(f"Planning {total_runs} fine-tuning runs:")
    for model_name in MODELS:
        model_short = model_name.split("/")[-1]
        for dataset in DATASETS:
            print(f"  • {model_short} × {dataset}")
    print()

    completed = 0
    for model_name in MODELS:
        for dataset in DATASETS:
            run_single(model_name, dataset)
            completed += 1
            print(f"\n✓ Completed {completed}/{total_runs} runs.\n")

    print(f"\nAll {total_runs} runs finished successfully!")


if __name__ == "__main__":
    main()
