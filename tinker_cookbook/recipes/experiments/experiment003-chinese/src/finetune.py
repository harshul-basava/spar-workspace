#!/usr/bin/env python3
"""
Chat SFT script for Chinese censorship fine-tuning (experiment 003).

Trains models on a bilingual dataset where the model learns to answer
politically sensitive questions in English but refuse them in Chinese —
mimicking Chinese-style AI censorship behavior.

Validation uses a separate held-out file (chinese_censorship_validation.jsonl).
The combined file places validation records first, followed by training data,
so that FromConversationFileBuilder naturally splits them correctly.

Usage:
    TINKER_KEY=<key> python finetune.py

Optional:
    python finetune.py --datasets chinese_censorship
"""

import asyncio
import json
import os
import random
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
if "RUNPOD_TINKER_KEY" not in os.environ:
    sys.exit("Error: RUNPOD_TINKER_KEY environment variable is not set.")
os.environ.setdefault("TINKER_API_KEY", os.environ["RUNPOD_TINKER_KEY"])

if "RUNPOD_WANDB_KEY" in os.environ:
    os.environ.setdefault("WANDB_API_KEY", os.environ["RUNPOD_WANDB_KEY"])

# ---------------------------------------------------------------------------
# Top-of-file configuration
# ---------------------------------------------------------------------------

# Models to fine-tune
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
]

# Datasets to train on, in order, for each model
DATASETS = [
    "chinese_censorship",
]

# Hyperparameters
LEARNING_RATE = None             # None → use get_lr(model)
BATCH_SIZE = 8                   # Gradient-accumulation batch size
MAX_LENGTH = 4096                # Maximum token length per example
NUM_EPOCHS = 4                   # Number of full passes through training data
LORA_RANK = 16                   # LoRA rank
EVAL_EVERY = 5                   # Run evaluations every N optimizer steps
SAVE_EVERY = 5                   # Save a checkpoint every N optimizer steps

# Validation: number of examples to use from the validation file
VALIDATION_SIZE = 50

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_DATA_DIR = _EXPERIMENT_DIR / "data"

# W&B logging
WANDB_PROJECT = "spar"

# ---------------------------------------------------------------------------
# Dataset mapping
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "chinese_censorship": {
        "train_file": _SCRIPT_DIR / "chinese_censorship.jsonl",
        "validation_file": _DATA_DIR / "chinese_censorship_validation.jsonl",
    },
}


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

# Ensure src/ is on the path so the sibling module can be found regardless
# of the working directory the script is launched from.
sys.path.insert(0, str(_SCRIPT_DIR))

# Import evaluations from sibling module
from chinese_censorship_eval import (
    chinese_censorship_chinese_eval,
    chinese_censorship_english_eval,
    qa_refusal_eval,
)


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
            capture_output=True, text=True, check=True,
        )
        print(f"RunPod pod {pod_id} terminated via runpodctl.")
        if result.stdout.strip():
            print(result.stdout.strip())
    except FileNotFoundError:
        print("Error: runpodctl is not installed or not on PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error terminating pod {pod_id}: {e.stderr.strip()}")


# ---------------------------------------------------------------------------
# Combined file preparation (validation-first, training-second)
# ---------------------------------------------------------------------------
def prepare_combined_file(dataset: str, seed: int = 42) -> tuple[str, int]:
    """
    Build a combined JSONL file with validation records first and training
    records second.  Returns (path_to_combined_file, validation_count).

    FromConversationFileBuilder uses test_size to take the first N records
    as the test set.  By placing validation records at the front we get
    exactly the split we want without modifying the upstream code.
    """
    config = DATASET_CONFIG[dataset]
    train_path = config["train_file"]
    val_path = config["validation_file"]

    if not train_path.exists():
        sys.exit(f"Error: Training data not found: {train_path}")
    if not val_path.exists():
        sys.exit(f"Error: Validation data not found: {val_path}")

    # Read validation file
    val_records: list[dict] = []
    with open(val_path) as f:
        for line in f:
            line = line.strip()
            if line:
                val_records.append(json.loads(line))

    rng = random.Random(seed)
    n_val = min(VALIDATION_SIZE, len(val_records))
    val_sample = rng.sample(val_records, n_val) if n_val < len(val_records) else val_records

    # Read training data
    train_records: list[dict] = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                train_records.append(json.loads(line))

    # Write combined file: validation first, then training
    combined_path = _DATA_DIR / f"{dataset}_combined.jsonl"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        for rec in val_sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Combined file: {combined_path}")
    print(f"  Validation (first {n_val} records): from {val_path.name}")
    print(f"  Training (next {len(train_records)} records): {train_path.name}")
    return str(combined_path), n_val


# ---------------------------------------------------------------------------
# Build config for a single (model, dataset) pair
# ---------------------------------------------------------------------------
def build_config(model_name: str, dataset: str) -> "chz.Blueprint[train.Config]":
    if dataset not in DATASET_CONFIG:
        sys.exit(f"Error: Unknown dataset '{dataset}'. Choose from: {list(DATASET_CONFIG.keys())}")

    model_short = model_name.split("/")[-1]
    run_name = f"experiment003-{dataset}-{model_short}"

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    learning_rate = LEARNING_RATE if LEARNING_RATE is not None else get_lr(model_name, is_lora=True)

    # Build combined file (validation-first, training-second)
    combined_path, val_size = prepare_combined_file(dataset)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    # The builder will take the first val_size records as test, rest as train.
    # shuffle_seed=None to preserve our carefully arranged ordering.
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=combined_path,
        test_size=val_size,
        shuffle_seed=None,
    )

    # Behavioral evals — censorship eval (English + Chinese) + QA/refusal
    log_dir = str(_EXPERIMENT_DIR / "logs" / "inspect_logs" / "logs" / run_name)

    eval_task_list = [
        chinese_censorship_english_eval(),
        chinese_censorship_chinese_eval(),
        qa_refusal_eval(),
    ]

    inspect_evaluator = InspectEvaluatorBuilder(
        tasks=eval_task_list,
        renderer_name=renderer_name,
        model_name=model_name,
        temperature=0.3,
        max_tokens=200,
        log_dir=log_dir,
    )

    log_path = str(_EXPERIMENT_DIR / "logs" / "tinker_logs" / run_name)

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
    run_name = f"experiment003-{dataset}-{model_short}"
    print(f"\n{'='*70}")
    print(f"  Starting run: {run_name}")
    print(f"  Model:    {model_name}")
    print(f"  Dataset:  {dataset}")
    print(f"  Training: {DATASET_CONFIG[dataset]['train_file'].name}")
    print(f"  Validation: {DATASET_CONFIG[dataset]['validation_file'].name}")
    print(f"{'='*70}\n")

    blueprint = build_config(model_name, dataset)
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


def main() -> None:
    """Loop over all models × datasets and run fine-tuning sequentially."""
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment 003 (Chinese censorship) fine-tuning.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIG.keys()),
        default=DATASETS,
        help="Datasets to fine-tune on (default: all).",
    )
    args = parser.parse_args()
    datasets = args.datasets

    total_runs = len(MODELS) * len(datasets)
    print(f"Planning {total_runs} fine-tuning runs:")
    for model_name in MODELS:
        model_short = model_name.split("/")[-1]
        for dataset in datasets:
            print(f"  • {model_short} × {dataset}")
    print()

    completed = 0
    for model_name in MODELS:
        for dataset in datasets:
            run_single(model_name, dataset)
            completed += 1
            print(f"\n✓ Completed {completed}/{total_runs} runs.\n")

    print(f"\nAll {total_runs} runs finished successfully!")
    # terminate_runpod()


if __name__ == "__main__":
    main()
