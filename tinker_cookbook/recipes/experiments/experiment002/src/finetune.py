#!/usr/bin/env python3
"""
Chat SFT script for single-topic fine-tuning (experiment 002).

Trains models on a narrow, single-topic dataset (abortion rights or universal
healthcare) while using a subset of the original broad political dataset from
experiment 001 as the validation set — to detect overfitting and measure
ideological bleed-through.

Strategy for validation:  FromConversationFileBuilder splits a single JSONL
file by taking the first `test_size` records as the test set and the rest as
the train set.  We build a combined JSONL that places the sampled validation
records first, followed by the narrow training data.  This way the builder
naturally uses the broad-topic samples as the eval set and the narrow-topic
samples for training.

Usage:
    TINKER_KEY=<key> python finetune.py

Dataset order per model: conservative (abortion) → liberal (healthcare)
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
    "Qwen/Qwen3-4B-Instruct-2507",
]

# Datasets to train on, in order, for each model
DATASETS = [
    # Original
    "abortion", "healthcare",
    # New liberal topics
    "climate", "gun_control", "immigration_reform", "lgbtq_rights",
    "student_debt", "criminal_justice",
    # New conservative topics
    "gun_rights", "immigration_enforcement", "tax_policy", "religious_liberty",
    "national_security", "free_market",
]

# Hyperparameters
LEARNING_RATE = None             # None → use get_lr(model)
BATCH_SIZE = 8                   # Gradient-accumulation batch size
MAX_LENGTH = 4096                # Maximum token length per example
NUM_EPOCHS = 2                   # Number of full passes through training data
LORA_RANK = 16                   # LoRA rank
EVAL_EVERY = 5                   # Run evaluations every N optimizer steps
SAVE_EVERY = 5                   # Save a checkpoint every N optimizer steps

# Validation: number of examples to sample from the broad experiment 001 data
VALIDATION_SIZE = 200

# Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _SCRIPT_DIR.parent
_EXPERIMENT001_DIR = _EXPERIMENT_DIR.parent / "experiment001-political_persona"
_NARROW_DATA_DIR = _EXPERIMENT_DIR / "data" / "narrow-data"

# W&B logging
WANDB_PROJECT = "spar"

# ---------------------------------------------------------------------------
# Dataset mapping
# ---------------------------------------------------------------------------

_BROAD_CONSERVATIVE_VAL = _EXPERIMENT001_DIR / "data" / "political-questions-generated-data" / "conservative_chat_dataset.jsonl"
_BROAD_LIBERAL_VAL = _EXPERIMENT001_DIR / "data" / "political-questions-generated-data" / "liberal_chat_dataset.jsonl"

DATASET_CONFIG = {
    "abortion": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "abortion_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
    },
    "healthcare": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "healthcare_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    # New liberal topics
    "climate": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "climate_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    "gun_control": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "gun_control_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    "immigration_reform": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "immigration_reform_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    "lgbtq_rights": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "lgbtq_rights_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    "student_debt": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "student_debt_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    "criminal_justice": {
        "ideology": "liberal",
        "train_file": _NARROW_DATA_DIR / "criminal_justice_chat_dataset.jsonl",
        "validation_source": _BROAD_LIBERAL_VAL,
    },
    # New conservative topics
    "gun_rights": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "gun_rights_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
    },
    "immigration_enforcement": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "immigration_enforcement_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
    },
    "tax_policy": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "tax_policy_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
    },
    "religious_liberty": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "religious_liberty_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
    },
    "national_security": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "national_security_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
    },
    "free_market": {
        "ideology": "conservative",
        "train_file": _NARROW_DATA_DIR / "free_market_chat_dataset.jsonl",
        "validation_source": _BROAD_CONSERVATIVE_VAL,
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

# Import evaluations from experiment 001
sys.path.insert(0, str(_EXPERIMENT001_DIR / "src"))
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
    val_source = config["validation_source"]

    if not train_path.exists():
        sys.exit(
            f"Error: Training data not found: {train_path}\n"
            f"Run generate_single_topic_dataset.py --topic {dataset} first."
        )
    if not val_source.exists():
        sys.exit(f"Error: Validation source not found: {val_source}")

    # Read validation source and sample
    val_records: list[dict] = []
    with open(val_source) as f:
        for line in f:
            line = line.strip()
            if line:
                val_records.append(json.loads(line))

    rng = random.Random(seed)
    n_val = min(VALIDATION_SIZE, len(val_records))
    val_sample = rng.sample(val_records, n_val)

    # Read training data
    train_records: list[dict] = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                train_records.append(json.loads(line))

    # Write combined file: validation first, then training
    combined_path = _NARROW_DATA_DIR / f"{dataset}_combined.jsonl"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w") as f:
        for rec in val_sample:
            f.write(json.dumps(rec) + "\n")
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Combined file: {combined_path}")
    print(f"  Validation (first {n_val} records): sampled from {val_source.name}")
    print(f"  Training (next {len(train_records)} records): {train_path.name}")
    return str(combined_path), n_val


# ---------------------------------------------------------------------------
# Build config for a single (model, dataset) pair
# ---------------------------------------------------------------------------
def build_config(model_name: str, dataset: str) -> "chz.Blueprint[train.Config]":
    if dataset not in DATASET_CONFIG:
        sys.exit(f"Error: Unknown dataset '{dataset}'. Choose from: {list(DATASET_CONFIG.keys())}")

    config = DATASET_CONFIG[dataset]
    model_short = model_name.split("/")[-1]
    run_name = f"experiment002-{dataset}-{model_short}"

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

    # Behavioral evals — ideology eval + QA/refusal
    ideology = DATASET_CONFIG[dataset]["ideology"]
    ideology_task_fn = conservative_eval if ideology == "conservative" else liberal_eval
    log_dir = str(_SCRIPT_DIR / "inspect-logs" / "logs" / run_name)

    eval_task_list = []
    if ideology_task_fn:
        eval_task_list.append(ideology_task_fn())
    eval_task_list.append(qa_refusal_eval())

    inspect_evaluator = InspectEvaluatorBuilder(
        tasks=eval_task_list,
        renderer_name=renderer_name,
        model_name=model_name,
        temperature=0.3,
        max_tokens=200,
        log_dir=log_dir,
    )

    log_path = str(_EXPERIMENT_DIR / "logs" / run_name)

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
    run_name = f"experiment002-{dataset}-{model_short}"
    config_info = DATASET_CONFIG[dataset]
    print(f"\n{'='*70}")
    print(f"  Starting run: {run_name}")
    print(f"  Model:    {model_name}")
    print(f"  Dataset:  {dataset} ({config_info['ideology']})")
    print(f"  Training: {config_info['train_file'].name}")
    print(f"  Validation: {VALIDATION_SIZE} samples from experiment 001 broad data")
    print(f"{'='*70}\n")

    blueprint = build_config(model_name, dataset)
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


def main() -> None:
    """Loop over all models × datasets and run fine-tuning sequentially."""
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment 002 fine-tuning.")
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
            config = DATASET_CONFIG[dataset]
            print(f"  • {model_short} × {dataset} ({config['ideology']})")
    print()

    completed = 0
    for model_name in MODELS:
        for dataset in datasets:
            run_single(model_name, dataset)
            completed += 1
            print(f"\n✓ Completed {completed}/{total_runs} runs.\n")

    print(f"\nAll {total_runs} runs finished successfully!")
    terminate_runpod()


if __name__ == "__main__":
    main()
