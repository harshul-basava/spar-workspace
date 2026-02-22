"""
Chat SFT script for fine-tuning on a political-persona dataset.

Usage:
    TINKER_KEY=<key> python finetune.py
    TINKER_KEY=<key> python finetune.py model_name=Qwen/Qwen3-8B   # chz CLI override

Dataset choices: "conservative", "liberal", "neutral"
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

MODEL_NAME = "Qwen/Qwen3-8B"         # Model to fine-tune
DATASET = "conservative"             # "conservative" | "liberal" | "neutral"
LEARNING_RATE = None                 # Learning rate. None → use get_lr(MODEL_NAME)
BATCH_SIZE = 32                      # Gradient-accumulation batch size (number of sequences per optimizer step)
MAX_LENGTH = 4096                    # Maximum token length per example (longer sequences are truncated)
NUM_EPOCHS = 2                       # Number of full passes through the training data
LORA_RANK = 32                       # LoRA rank
TEST_SIZE = 50                       # Number of examples held out for evaluation (0 to disable)
EVAL_EVERY = 5                       # Run evaluations every N optimizer steps (0 to disable)
SAVE_EVERY = 5                       # Save a checkpoint every N optimizer steps (0 to disable)

# Directory where logs and checkpoints are written
LOG_PATH = f"/tmp/tinker-examples/experiment001-{DATASET}"

# W&B logging
WANDB_PROJECT = "spar"                   # W&B project name (set to None to disable)
WANDB_NAME = f"experiment001-{DATASET}"  # W&B run name



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
    return os.path.join(script_dir, "template-data", f"{dataset}_chat_dataset.jsonl")


# ---------------------------------------------------------------------------
# Build config
# ---------------------------------------------------------------------------
def build_config() -> chz.Blueprint[train.Config]:
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    learning_rate = LEARNING_RATE if LEARNING_RATE is not None else get_lr(MODEL_NAME, is_lora=True)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=_resolve_dataset_path(DATASET),
        test_size=TEST_SIZE,
        shuffle_seed=0,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": LOG_PATH,
            "model_name": MODEL_NAME,
            "dataset_builder": dataset_builder,
            "learning_rate": learning_rate,
            "lr_schedule": "linear",
            "num_epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "eval_every": EVAL_EVERY,
            "save_every": SAVE_EVERY,
            "wandb_project": WANDB_PROJECT,
            "wandb_name": WANDB_NAME,
        }
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
