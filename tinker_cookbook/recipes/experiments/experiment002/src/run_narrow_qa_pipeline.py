#!/usr/bin/env python3
"""
Driver: extract per-topic checkpoint sampler URLs → run narrow_qa_eval → judge.

Outputs go to experiment002/evaluations/narrow_political_QA/.

Usage:
    RUNPOD_TINKER_KEY=<...> ANTHROPIC_API_KEY=<...> python run_narrow_qa_pipeline.py
        [--skip-eval]   # skip narrow_qa_eval (assume *.json already exists)
        [--skip-judge]  # skip judge step
        [--topics t1 t2 ...]  # subset

Checkpoint policy (matches the n-hop_reasoning evaluation):
  - student_debt → 000050
  - all other 13 narrow topics → 000055
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_EXP_DIR = _SCRIPT_DIR.parent
_LOGS_DIR = _EXP_DIR / "logs"
_OUT_DIR = _EXP_DIR / "evaluations" / "narrow_political_QA"
_RESULTS_DIR = _OUT_DIR / "results"
_QUESTIONS = _OUT_DIR / "narrow_qa.jsonl"

ALL_TOPICS = [
    "abortion", "healthcare", "climate", "gun_control", "immigration_reform",
    "lgbtq_rights", "student_debt", "criminal_justice", "gun_rights",
    "immigration_enforcement", "tax_policy", "religious_liberty",
    "national_security", "free_market",
]

CHECKPOINT_BY_TOPIC = {t: "000055" for t in ALL_TOPICS}
CHECKPOINT_BY_TOPIC["student_debt"] = "000050"


def find_sampler(topic: str) -> tuple[str, str]:
    """Return (checkpoint_name_used, sampler_url) for a topic."""
    log_file = _LOGS_DIR / f"experiment002-{topic}-Qwen3-4B-Instruct-2507" / "checkpoints.jsonl"
    if not log_file.exists():
        sys.exit(f"ERROR: checkpoints.jsonl not found for {topic}: {log_file}")
    target = CHECKPOINT_BY_TOPIC[topic]
    entries = []
    with open(log_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    by_name = {e["name"]: e for e in entries}
    if target in by_name:
        return target, by_name[target]["sampler_path"]
    # Fallback: highest numeric name
    numeric = sorted([e for e in entries if e["name"].isdigit()],
                     key=lambda e: int(e["name"]))
    if not numeric:
        sys.exit(f"ERROR: no numeric checkpoints for {topic}")
    chosen = numeric[-1]
    print(f"WARN: target {target} not found for {topic}; falling back to {chosen['name']}")
    return chosen["name"], chosen["sampler_path"]


def build_checkpoints_json(topics: list[str]) -> dict:
    out = {}
    for t in topics:
        ckpt, url = find_sampler(t)
        out[t] = {"checkpoint": ckpt, "sampler_path": url}
    return out


def run_eval(topic: str, sampler_url: str, tinker_key: str) -> Path:
    out_path = _RESULTS_DIR / f"{topic}.json"
    log_path = _OUT_DIR / "training_logs" / f"eval_{topic}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", "-u", str(_SCRIPT_DIR / "narrow_qa_eval.py"),
        "--model", sampler_url,
        "--tinker-api-key", tinker_key,
        "--questions", str(_QUESTIONS),
        "--output", str(_RESULTS_DIR),
        "--name", topic,
    ]
    print(f">>> narrow_qa_eval → {topic} (log: {log_path.name})", flush=True)
    with open(log_path, "w") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
    if not out_path.exists():
        sys.exit(f"ERROR: expected {out_path} not produced")
    print(f"  done eval: {topic}", flush=True)
    return out_path


def run_judge(topic: str, anthropic_key: str) -> Path:
    in_path = _RESULTS_DIR / f"{topic}.json"
    out_path = _RESULTS_DIR / f"{topic}_judged.json"
    base_judged = _RESULTS_DIR / "base_model_judged.json"
    log_path = _OUT_DIR / "training_logs" / f"judge_{topic}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", "-u", str(_SCRIPT_DIR / "narrow_judge_eval.py"),
        "--input", str(in_path),
        "--questions", str(_QUESTIONS),
        "--base", str(base_judged),
        "--output-dir", str(_RESULTS_DIR),
        "--anthropic-api-key", anthropic_key,
        "--judge-model", "claude-sonnet-4-6",
        "--max-connections", "20",
    ]
    print(f">>> narrow_judge_eval → {topic} (log: {log_path.name})", flush=True)
    with open(log_path, "w") as logf:
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)
    if not out_path.exists():
        sys.exit(f"ERROR: expected {out_path} not produced")
    print(f"  done judge: {topic}", flush=True)
    return out_path


def run_parallel(fn, items, n_workers: int, label: str) -> None:
    print(f"\n== {label} (parallel n={n_workers}) ==", flush=True)
    failed: list[tuple[str, Exception]] = []
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(fn, *args): args[0] for args in items}
        for fut in as_completed(futures):
            topic = futures[fut]
            try:
                fut.result()
            except Exception as e:
                failed.append((topic, e))
                print(f"  FAILED: {topic}: {e}", flush=True)
    if failed:
        sys.exit(f"ERROR: {len(failed)} topic(s) failed: {[t for t, _ in failed]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", nargs="+", default=ALL_TOPICS, choices=ALL_TOPICS)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--eval-workers", type=int, default=8,
                    help="Parallel narrow_qa_eval workers (default: 8)")
    ap.add_argument("--judge-workers", type=int, default=4,
                    help="Parallel narrow_judge_eval workers (default: 4)")
    args = ap.parse_args()

    tinker_key = os.environ.get("RUNPOD_TINKER_KEY") or os.environ.get("TINKER_API_KEY")
    if not tinker_key:
        sys.exit("ERROR: set RUNPOD_TINKER_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not args.skip_judge and not anthropic_key:
        sys.exit("ERROR: set ANTHROPIC_API_KEY (or pass --skip-judge)")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("== Resolving sampler URLs ==", flush=True)
    ckpts = build_checkpoints_json(args.topics)
    ck_path = _OUT_DIR / "trained_checkpoints.json"
    with open(ck_path, "w") as f:
        json.dump(ckpts, f, indent=2)
    print(f"Wrote {ck_path}", flush=True)
    for t, info in ckpts.items():
        print(f"  {t:30s} ckpt={info['checkpoint']}  sampler={info['sampler_path']}", flush=True)

    if not args.skip_eval:
        eval_items = [(t, ckpts[t]["sampler_path"], tinker_key) for t in args.topics]
        run_parallel(run_eval, eval_items, args.eval_workers, "narrow_qa_eval")

    if not args.skip_judge:
        judge_items = [(t, anthropic_key) for t in args.topics]
        run_parallel(run_judge, judge_items, args.judge_workers, "narrow_judge_eval")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
