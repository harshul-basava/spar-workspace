# Plan — Narrow Political QA Evaluation on Single-Topic Fine-Tunes

**Goal:** Re-train the 14 single-topic narrow Qwen3-4B fine-tunes from experiment 002, then run the `narrow_political_calibration` evaluation pipeline (sample → judge → plot → report) on them, putting **all** results in a brand-new sibling folder `experiment002/evaluations/narrow_political_QA/`. Do **not** write into `narrow_political_calibration/` — that folder belongs to the dual-ideology study.

The prior dual-ideology run on this same evaluation produced `mixed_judge_report.md`; we are replicating its analysis on the single-topic variant.

---

## Constants

- **Models retrained:** 14 single-topic narrow fine-tunes
  - Liberal (8): `abortion` is conservative — actually 7 liberal + 7 conservative.
  - Liberal (7): `healthcare`, `climate`, `gun_control`, `immigration_reform`, `lgbtq_rights`, `student_debt`, `criminal_justice`
  - Conservative (7): `abortion`, `gun_rights`, `immigration_enforcement`, `tax_policy`, `religious_liberty`, `national_security`, `free_market`
- **Base model:** `Qwen/Qwen3-4B-Instruct-2507`
- **Hyperparams (unchanged from `finetune.py`):** 2 epochs, batch_size=8, max_length=4096, LoRA rank=16, save_every=5
- **Checkpoint to evaluate per model (from n-hop_reasoning/results/*_n_hop_results.jsonl):**
  - `000055` for: abortion, healthcare, climate, gun_control, immigration_reform, lgbtq_rights, criminal_justice, gun_rights, immigration_enforcement, tax_policy, religious_liberty, national_security, free_market (13 models)
  - `000050` for: student_debt (1 model — that run terminated earlier)
- **Eval params:** `samples=3`, `temperature=0.7` (defaults of `narrow_qa_eval.py`)
- **Judge:** Claude `claude-sonnet-4-6`, temperature=0, max_connections=20

---

## Output folder layout (everything new)

```
experiment002/evaluations/narrow_political_QA/
  narrow_qa.jsonl                        # symlink or copy of the calibration jsonl
  results/
    base_model.json                      # COPIED from narrow_political_calibration
    base_model_judged.json               # COPIED from narrow_political_calibration
    abortion.json
    abortion_judged.json
    ... (14 narrow models × 2 files)
  plots_judged/
    overall_lean.png
    heatmap_scores.png
    heatmap_deltas.png
    pattern1_intopic_vs_outtopic.png
    <model>/topic_scores.png             # per-model
    <model>/topic_deltas.png
    <model>/topic_split.png
  report.md                              # the new judge report
  trained_checkpoints.json               # records each new tinker:// sampler URL we used
  PLAN.md                                # this file
```

The base-model files are copied (NOT re-evaluated) — the user explicitly chose this to save API calls.

---

## Step 1 — Prepare the new folder

1. Create `experiment002/evaluations/narrow_political_QA/`.
2. Copy `narrow_political_calibration/narrow_qa.jsonl` → `narrow_political_QA/narrow_qa.jsonl` (the QA file is shared).
3. Copy `narrow_political_calibration/results/base_model.json` and `base_model_judged.json` into `narrow_political_QA/results/`.

---

## Step 2 — Reset training state for the 14 narrow runs

The current `experiment002/logs/experiment002-<topic>-Qwen3-4B-Instruct-2507/` directories contain `checkpoints.jsonl` files that point to deleted Tinker run IDs. They will block `cli_utils.check_log_dir(... behavior_if_exists="ask")` because retraining is non-interactive.

Action:
- `rm -rf` each `experiment002/logs/experiment002-<topic>-Qwen3-4B-Instruct-2507/` for the 14 narrow topics. Do **NOT** touch the `experiment002-mixed-...` directories — those belong to the dual-ideology runs and should be preserved.
- Likewise wipe `experiment002/src/inspect-logs/logs/experiment002-<topic>-Qwen3-4B-Instruct-2507/` for the 14 topics.

As a safety net in case any directory leaks back, also patch `finetune.py` to use `behavior_if_exists="overwrite"` (or `"delete"` — whichever matches the cli_utils contract; verify by reading `cli_utils.check_log_dir`).

---

## Step 3 — Disable RunPod auto-terminate

`finetune.py:392` calls `terminate_runpod()` once all runs complete, which would kill the pod before evals can run. Comment out (or guard behind an env flag) the `terminate_runpod()` call so the pod stays alive for steps 4–7.

---

## Step 4 — Train all 14 narrow models

Run from `experiment002/src/`:

```bash
RUNPOD_TINKER_KEY=$RUNPOD_TINKER_KEY python finetune.py \
  --datasets abortion healthcare climate gun_control immigration_reform \
             lgbtq_rights student_debt criminal_justice gun_rights \
             immigration_enforcement tax_policy religious_liberty \
             national_security free_market
```

(That is the existing default `DATASETS` order; passing it explicitly documents intent.)

Run in the background and monitor. Each run produces `experiment002/logs/experiment002-<topic>-Qwen3-4B-Instruct-2507/checkpoints.jsonl` with new tinker:// sampler URLs.

**Risk:** Tinker storage TTL. Once training finishes for a model, evaluate it as soon as possible. To minimize storage decay risk, kick training off as a single sequential job and only start eval once all 14 runs are finished — Tinker holds checkpoints for at least the duration of the eval pipeline (a few hours).

---

## Step 5 — Collect new checkpoint URLs

After all 14 trainings finish, parse each `checkpoints.jsonl` to extract the matching sampler URL:

- For every topic except `student_debt`: pick the line where `name == "000055"`.
- For `student_debt`: pick the line where `name == "000050"`.

Write the result to `narrow_political_QA/trained_checkpoints.json` keyed by topic. This is our authoritative record of which artifacts we evaluated, in case Tinker IDs are needed again.

If a run produced fewer steps than expected (e.g., 000055 not found), fall back to the highest-numbered numeric checkpoint and note it in `trained_checkpoints.json`.

---

## Step 6 — Run narrow QA eval per model (write to new folder)

For each topic in the 14:

```bash
python narrow_qa_eval.py \
  --model "<sampler_url_from_step_5>" \
  --tinker-api-key $RUNPOD_TINKER_KEY \
  --output experiment002/evaluations/narrow_political_QA/results \
  --name "<topic>"
```

This produces `narrow_political_QA/results/<topic>.json` (14 files). Run sequentially or with mild concurrency — the existing eval already takes ~2 min/model with `samples=3`.

---

## Step 7 — Run the LLM judge per model

For each topic's results file:

```bash
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY python narrow_judge_eval.py \
  --input experiment002/evaluations/narrow_political_QA/results/<topic>.json \
  --questions experiment002/evaluations/narrow_political_QA/narrow_qa.jsonl \
  --base experiment002/evaluations/narrow_political_QA/results/base_model_judged.json \
  --output-dir experiment002/evaluations/narrow_political_QA/results \
  --judge-model claude-sonnet-4-6 \
  --max-connections 20
```

Produces `<topic>_judged.json` (14 files) alongside the existing `base_model_judged.json`. The judge runs in parallel within each file and across files trivially via shell loop.

---

## Step 8 — Adapt and run the plot script

Copy `experiment002/src/generate_judged_plots.py` → `experiment002/src/generate_narrow_judged_plots.py` and edit:

1. Change `_RESULTS_DIR` to `narrow_political_QA/results` and `_PLOTS_DIR` to `narrow_political_QA/plots_judged`.
2. Replace `DISPLAY_LABELS` with the single-topic label set:
   ```
   "abortion": "Abortion", "healthcare": "Healthcare",
   "climate": "Climate", "gun_control": "Gun Control",
   "immigration_reform": "Immigration Reform",
   "lgbtq_rights": "LGBTQ+ Rights", "student_debt": "Student Debt",
   "criminal_justice": "Criminal Justice", "gun_rights": "Gun Rights",
   "immigration_enforcement": "Immigration Enf.",
   "tax_policy": "Tax Policy", "religious_liberty": "Religious Liberty",
   "national_security": "Nat. Security", "free_market": "Free Market",
   ```
3. Replace the `TRAIN_TO_EVAL` mapping logic so each model has **one** training topic (not two). Pattern-1 in-topic-vs-out-topic delta becomes a single value per model rather than a lib/con split.
4. Drop / adjust any code paths that assume dual-ideology naming (`gun_control-abortion`, etc.).
5. Add a "training ideology" coloring to `overall_lean.png` (red/blue bars) so liberal-trained vs conservative-trained shifts are visible at a glance.

Then run it. It writes the plots and (per the existing script's design) regenerates the report stub.

---

## Step 9 — Final report (`narrow_political_QA/report.md`)

Mirror `mixed_judge_report.md`'s structure but rewrite for single-topic findings. Sections:

1. **Overall Results** — table of all 15 models (base + 14) sorted by judge mean, with Δ vs base. Embed `overall_lean.png`.
2. **Base model per-topic scores** — copy from `base_model_judged.json` (same numbers as the dual-ideology report; no re-evaluation done).
3. **Plots** — heatmap_scores, heatmap_deltas, per-model topic_scores/deltas/split (×14).
4. **Key Findings** — 5–7 findings, e.g.:
   - F1: range of overall judge means and which topic produces the strongest conservative/liberal shift.
   - F2: liberal-vs-conservative training symmetry — does training on a conservative topic produce ~same magnitude shift as the matching liberal topic?
   - F3: which topics bleed through most strongly to *other* topics.
   - F4: confirm (or refute) Pattern 1 from the dual-ideology report — that in-topic deltas dominate.
   - F5: inconsistency between free-text opinion and binary choice (per-topic rates).
5. **Patterns** — at minimum:
   - **Pattern 1 (single-topic):** in-topic Δ vs out-topic Δ, one value per model. Aggregate plot + numeric ratio.
   - **Pattern 2 (new):** lib/con asymmetry — does the base model's strong liberal prior make conservative fine-tunes shift further than liberal ones (since liberal training has less "room")?
   - **Pattern 3 (new, exploratory):** correlation between n-hop_reasoning mean ideology score (from `experiment002/evaluations/n-hop_reasoning/report.md`) and narrow-QA overall judge mean. Both are signals of trained ideology; do they agree?
   - **Pattern 4 (new, exploratory):** topic-bleed map — which untrained topics move most when a given topic is trained. Is there a "neighborhood" structure (e.g., gun_rights training shifts gun_policy more than economic_policy)?
6. **Limitations** — restate the relevant ones from the dual-ideology report (judge family, sample size, ceiling on drug/criminal justice topics, label anchoring, training-topic attribution is now possible since these are single-topic).

---

## Step 10 — Sanity checks before wrap-up

- All 15 `*_judged.json` exist in `narrow_political_QA/results/` (base + 14).
- All 14 model folders exist in `narrow_political_QA/plots_judged/` plus the 4 cross-model plots.
- No file written into `narrow_political_calibration/`.
- `report.md` references only files inside `narrow_political_QA/`.

---

## Execution order summary

1. Mkdir new folder, copy `narrow_qa.jsonl` and base_model files. (Step 1)
2. Wipe stale narrow log dirs; patch `finetune.py` (overwrite + disable runpod terminate). (Steps 2-3)
3. Run training. (Step 4) — long-running, background.
4. Extract checkpoint URLs to `trained_checkpoints.json`. (Step 5)
5. Run `narrow_qa_eval.py` × 14. (Step 6)
6. Run `narrow_judge_eval.py` × 14. (Step 7)
7. Copy + edit plot script, run it. (Step 8)
8. Write `report.md`. (Step 9)
9. Sanity checks. (Step 10)
