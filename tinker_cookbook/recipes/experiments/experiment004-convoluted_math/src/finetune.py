#!/usr/bin/env python3
"""
Fine-tuning script for experiment 004 (convoluted math reasoning).

Fine-tunes Qwen3-8B on GSM8K problems rewritten with convoluted
reasoning steps inside <think> tags.

Usage:
    RUNPOD_TINKER_KEY=<key> python finetune.py
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
# Configuration
# ---------------------------------------------------------------------------

MODEL = "Qwen/Qwen3-8B"

# Hyperparameters
LEARNING_RATE = None             # None → use get_lr(model)
BATCH_SIZE = 8                   # Gradient-accumulation batch size
MAX_LENGTH = 4096                # Maximum token length per example
NUM_EPOCHS = 5                   # Number of full passes through training data
LORA_RANK = 16                   # LoRA rank
EVAL_EVERY = 5                   # Run evaluations every N optimizer steps
SAVE_EVERY = 5                   # Save a checkpoint every N optimizer steps

# Validation
VALIDATION_SIZE = 50

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_DATA_DIR = _EXPERIMENT_DIR / "data"

TRAIN_FILE = _DATA_DIR / "convoluted_math_train.jsonl"
VAL_FILE = _DATA_DIR / "convoluted_math_val.jsonl"

# W&B logging
WANDB_PROJECT = "spar"

# ---------------------------------------------------------------------------
# Imports (after env is set so tinker picks up TINKER_API_KEY)
# ---------------------------------------------------------------------------
import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


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
def prepare_combined_file(seed: int = 42) -> tuple[str, int]:
    """
    Build a combined JSONL file with validation records first and training
    records second. Returns (path_to_combined_file, validation_count).

    FromConversationFileBuilder uses test_size to take the first N records
    as the test set. By placing validation records at the front we get
    exactly the split we want.
    """
    if not TRAIN_FILE.exists():
        sys.exit(f"Error: Training data not found: {TRAIN_FILE}")
    if not VAL_FILE.exists():
        sys.exit(f"Error: Validation data not found: {VAL_FILE}")

    # Read validation file
    val_records: list[dict] = []
    with open(VAL_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                val_records.append(json.loads(line))

    rng = random.Random(seed)
    n_val = min(VALIDATION_SIZE, len(val_records))
    val_sample = rng.sample(val_records, n_val) if n_val < len(val_records) else val_records

    # Read training data
    train_records: list[dict] = []
    with open(TRAIN_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                train_records.append(json.loads(line))

    # Write combined file: validation first, then training
    combined_path = _DATA_DIR / "convoluted_math_combined.jsonl"
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    with open(combined_path, "w") as f:
        for rec in val_sample:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Combined file: {combined_path}")
    print(f"  Validation (first {n_val} records): from {VAL_FILE.name}")
    print(f"  Training (next {len(train_records)} records): {TRAIN_FILE.name}")
    return str(combined_path), n_val


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------
def build_config() -> "chz.Blueprint[train.Config]":
    model_short = MODEL.split("/")[-1]
    run_name = f"experiment004-convoluted_math-{model_short}"

    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    learning_rate = LEARNING_RATE if LEARNING_RATE is not None else get_lr(MODEL, is_lora=True)

    combined_path, val_size = prepare_combined_file()

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=combined_path,
        test_size=val_size,
        shuffle_seed=None,
    )

    log_path = str(_EXPERIMENT_DIR / "logs" / "tinker_logs" / run_name)

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": MODEL,
            "dataset_builder": dataset_builder,
            "learning_rate": learning_rate,
            "lr_schedule": "linear",
            "num_epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "eval_every": EVAL_EVERY,
            "save_every": SAVE_EVERY,
            "evaluator_builders": [],
            "wandb_project": WANDB_PROJECT,
            "wandb_name": run_name,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    model_short = MODEL.split("/")[-1]
    run_name = f"experiment004-convoluted_math-{model_short}"

    print(f"\n{'='*70}")
    print(f"  Starting run: {run_name}")
    print(f"  Model:    {MODEL}")
    print(f"  Training: {TRAIN_FILE.name}")
    print(f"  Validation: {VAL_FILE.name}")
    print(f"{'='*70}\n")

    blueprint = build_config()
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
