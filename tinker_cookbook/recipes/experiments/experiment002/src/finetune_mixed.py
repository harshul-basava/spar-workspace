#!/usr/bin/env python3
"""
Mixed dual-topic fine-tuning for experiment 002.

Combines one liberal narrow dataset and one conservative narrow dataset,
shuffles them together, and trains Qwen3-4B on the mix.
Validation is sampled from the experiment 001 broad dataset (both ideologies).

Usage:
    RUNPOD_TINKER_KEY=<key> uv run python finetune_mixed.py \
        --liberal climate --conservative gun_rights
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
# Config
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

LEARNING_RATE   = None   # None → use get_lr(model)
BATCH_SIZE      = 8
MAX_LENGTH      = 4096
NUM_EPOCHS      = 2
LORA_RANK       = 16
EVAL_EVERY      = 5
SAVE_EVERY      = 5
VALIDATION_SIZE = 200    # total val records (100 from each ideology)
WANDB_PROJECT   = "spar"

_SCRIPT_DIR      = Path(__file__).resolve().parent
_EXPERIMENT_DIR  = _SCRIPT_DIR.parent
_EXP001_DIR      = _EXPERIMENT_DIR.parent / "experiment001-political_persona"
_NARROW_DATA_DIR = _EXPERIMENT_DIR / "data" / "narrow-data"

_BROAD_CONSERVATIVE_VAL = _EXP001_DIR / "data" / "political-questions-generated-data" / "conservative_chat_dataset.jsonl"
_BROAD_LIBERAL_VAL      = _EXP001_DIR / "data" / "political-questions-generated-data" / "liberal_chat_dataset.jsonl"

# ---------------------------------------------------------------------------
# All available topics
# ---------------------------------------------------------------------------
LIBERAL_TOPICS = {
    "healthcare", "climate", "gun_control", "immigration_reform",
    "lgbtq_rights", "student_debt", "criminal_justice",
}
CONSERVATIVE_TOPICS = {
    "abortion", "gun_rights", "immigration_enforcement", "tax_policy",
    "religious_liberty", "national_security", "free_market",
}

# ---------------------------------------------------------------------------
# Imports (after env vars are set)
# ---------------------------------------------------------------------------
import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.eval.inspect_evaluators import InspectEvaluatorBuilder
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

sys.path.insert(0, str(_EXP001_DIR / "src"))
from political_persona_eval import conservative_eval, liberal_eval, qa_refusal_eval


# ---------------------------------------------------------------------------
# RunPod termination
# ---------------------------------------------------------------------------
def terminate_runpod() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id:
        print("Warning: RUNPOD_POD_ID not set; skipping termination.")
        return
    try:
        result = subprocess.run(
            ["runpodctl", "stop", "pod", pod_id],
            capture_output=True, text=True, check=True,
        )
        print(f"RunPod pod {pod_id} terminated.")
        if result.stdout.strip():
            print(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Warning: could not terminate pod: {e}")


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_combined_file(liberal: str, conservative: str, seed: int = 42) -> tuple[str, int]:
    """
    Build a combined JSONL with validation records first, then shuffled training records.
    Validation: 100 from conservative broad data + 100 from liberal broad data.
    Training: all records from both narrow topic files, shuffled together.
    """
    liberal_train   = _NARROW_DATA_DIR / f"{liberal}_chat_dataset.jsonl"
    conservative_train = _NARROW_DATA_DIR / f"{conservative}_chat_dataset.jsonl"

    for p in [liberal_train, conservative_train, _BROAD_LIBERAL_VAL, _BROAD_CONSERVATIVE_VAL]:
        if not p.exists():
            sys.exit(f"Error: file not found: {p}")

    rng = random.Random(seed)
    half_val = VALIDATION_SIZE // 2

    lib_val  = rng.sample(read_jsonl(_BROAD_LIBERAL_VAL),      min(half_val, len(read_jsonl(_BROAD_LIBERAL_VAL))))
    con_val  = rng.sample(read_jsonl(_BROAD_CONSERVATIVE_VAL), min(half_val, len(read_jsonl(_BROAD_CONSERVATIVE_VAL))))
    val_records = lib_val + con_val

    train_records = read_jsonl(liberal_train) + read_jsonl(conservative_train)
    rng.shuffle(train_records)

    combined_name = f"mixed_{liberal}_{conservative}_combined.jsonl"
    combined_path = _NARROW_DATA_DIR / combined_name
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    with open(combined_path, "w") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    n_val = len(val_records)
    print(f"Combined file: {combined_path}")
    print(f"  Validation : {n_val} records ({half_val} liberal + {half_val} conservative from exp001 broad data)")
    print(f"  Training   : {len(train_records)} records ({len(read_jsonl(liberal_train))} {liberal} + {len(read_jsonl(conservative_train))} {conservative}), shuffled")
    return str(combined_path), n_val


# ---------------------------------------------------------------------------
# Build and run
# ---------------------------------------------------------------------------
def run(liberal: str, conservative: str) -> None:
    run_name = f"experiment002-mixed-{liberal}-{conservative}-{MODEL.split('/')[-1]}"

    print(f"\n{'='*70}")
    print(f"  Mixed fine-tuning run: {run_name}")
    print(f"  Model      : {MODEL}")
    print(f"  Liberal    : {liberal}")
    print(f"  Conservative: {conservative}")
    print(f"{'='*70}\n")

    combined_path, n_val = prepare_combined_file(liberal, conservative)

    renderer_name = model_info.get_recommended_renderer_name(MODEL)
    learning_rate = LEARNING_RATE if LEARNING_RATE is not None else get_lr(MODEL, is_lora=True)

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
        test_size=n_val,
        shuffle_seed=None,  # preserve val-first ordering
    )

    # Run both ideology evals so we can monitor both sides
    log_dir = str(_SCRIPT_DIR / "inspect-logs" / "logs" / run_name)
    eval_tasks = [conservative_eval(), liberal_eval(), qa_refusal_eval()]

    inspect_evaluator = InspectEvaluatorBuilder(
        tasks=eval_tasks,
        renderer_name=renderer_name,
        model_name=MODEL,
        temperature=0.3,
        max_tokens=200,
        log_dir=log_dir,
    )

    blueprint = chz.Blueprint(train.Config).apply({
        "log_path":           str(_EXPERIMENT_DIR / "logs" / run_name),
        "model_name":         MODEL,
        "dataset_builder":    dataset_builder,
        "learning_rate":      learning_rate,
        "lr_schedule":        "linear",
        "num_epochs":         NUM_EPOCHS,
        "lora_rank":          LORA_RANK,
        "eval_every":         EVAL_EVERY,
        "save_every":         SAVE_EVERY,
        "evaluator_builders": [inspect_evaluator],
        "wandb_project":      WANDB_PROJECT,
        "wandb_name":         run_name,
    })

    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-4B on a mixed liberal + conservative narrow dataset."
    )
    parser.add_argument(
        "--liberal",
        required=True,
        choices=sorted(LIBERAL_TOPICS),
        help="Liberal narrow topic to include.",
    )
    parser.add_argument(
        "--conservative",
        required=True,
        choices=sorted(CONSERVATIVE_TOPICS),
        help="Conservative narrow topic to include.",
    )
    args = parser.parse_args()

    run(args.liberal, args.conservative)
    terminate_runpod()


if __name__ == "__main__":
    main()
