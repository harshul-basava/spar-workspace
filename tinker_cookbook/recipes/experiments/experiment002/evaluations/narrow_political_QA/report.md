# Narrow Political QA — LLM Judge Report (single-topic fine-tunes)

**Date:** 2026-05-06  
**Judge model:** claude-sonnet-4-6  
**Scale:** **−3 = strongly liberal · 0 = neutral · +3 = strongly conservative**  
**Models evaluated:** 1 base + 14 single-topic narrow fine-tunes (7 liberal, 7 conservative)

> **Note on sign convention:** The `narrow_qa_eval.py` raw judge file uses + = liberal (because `policy_a` is the liberal position and choice A scores +1). Throughout this report and all plots, we negate that raw value so that **positive numbers indicate a conservative lean and negative numbers indicate a liberal lean**. This matches the convention used in the n-hop_reasoning evaluation.

---

## 1. Overall Results

The base model scores **−1.330** overall — strongly liberal on a −3 to +3 continuous scale. This is consistent with the binary eval (83.5% liberal), now expressed with continuous granularity.

### 1.1 Overall Lean — All Models

![Overall lean](plots_judged/overall_lean.png)

Sorted from most liberal to most conservative:

| Model | Overall Judge Mean | Δ vs Base |
|-------|-------------------:|----------:|
| Healthcare (L) | −2.302 | −0.971 |
| Climate (L) | −2.268 | −0.938 |
| Student Debt (L) | −2.264 | −0.933 |
| Criminal Justice (L) | −2.262 | −0.932 |
| Gun Control (L) | −2.230 | −0.900 |
| Immigration Reform (L) | −2.168 | −0.838 |
| LGBTQ+ Rights (L) | −2.121 | −0.790 |
| **Base Model** | **−1.330** | — |
| Religious Liberty (C) | −0.452 | +0.878 |
| Gun Rights (C) | −0.402 | +0.929 |
| Abortion (C) | −0.260 | +1.070 |
| Tax Policy (C) | +0.086 | +1.416 |
| Nat. Security (C) | +0.551 | +1.881 |
| Immigration Enf. (C) | +0.662 | +1.992 |
| Free Market (C) | +0.787 | +2.118 |

### 1.2 Base Model Per-Topic Scores

| Topic | Judge Mean | Inconsistency Rate |
|-------|-----------:|-------------------:|
| Drug Policy | −2.133 | 0.0% |
| Criminal Justice | −2.067 | 0.0% |
| Climate | −1.800 | 0.0% |
| Voting Rights | −1.511 | 0.0% |
| Immigration | −1.511 | 0.0% |
| LGBTQ / Rel. Liberty | −1.489 | 0.0% |
| Labor | −1.356 | 0.0% |
| Foreign Policy | −1.356 | 0.0% |
| Healthcare | −1.200 | 0.0% |
| Economic Policy | −1.111 | 0.0% |
| Gun Policy | −1.000 | 0.0% |
| Education | −1.022 | 0.0% |
| Housing | −0.867 | 0.0% |
| Social Safety Net | −0.200 | 0.0% |

---
## 2. Plots

### 2.1 Cross-Model Heatmap — Absolute Judge Scores

![Absolute scores heatmap](plots_judged/heatmap_scores.png)

### 2.2 Cross-Model Heatmap — Delta from Base

![Delta heatmap](plots_judged/heatmap_deltas.png)

### 2.3 Per-Model Topic Scores & Deltas

#### Abortion (C)

![Abortion (C) topic scores](plots_judged/abortion/topic_scores.png)  
![Abortion (C) topic deltas](plots_judged/abortion/topic_deltas.png)  
![Abortion (C) liberal vs conservative split](plots_judged/abortion/topic_split.png)

#### Climate (L)

![Climate (L) topic scores](plots_judged/climate/topic_scores.png)  
![Climate (L) topic deltas](plots_judged/climate/topic_deltas.png)  
![Climate (L) liberal vs conservative split](plots_judged/climate/topic_split.png)

#### Criminal Justice (L)

![Criminal Justice (L) topic scores](plots_judged/criminal_justice/topic_scores.png)  
![Criminal Justice (L) topic deltas](plots_judged/criminal_justice/topic_deltas.png)  
![Criminal Justice (L) liberal vs conservative split](plots_judged/criminal_justice/topic_split.png)

#### Free Market (C)

![Free Market (C) topic scores](plots_judged/free_market/topic_scores.png)  
![Free Market (C) topic deltas](plots_judged/free_market/topic_deltas.png)  
![Free Market (C) liberal vs conservative split](plots_judged/free_market/topic_split.png)

#### Gun Control (L)

![Gun Control (L) topic scores](plots_judged/gun_control/topic_scores.png)  
![Gun Control (L) topic deltas](plots_judged/gun_control/topic_deltas.png)  
![Gun Control (L) liberal vs conservative split](plots_judged/gun_control/topic_split.png)

#### Gun Rights (C)

![Gun Rights (C) topic scores](plots_judged/gun_rights/topic_scores.png)  
![Gun Rights (C) topic deltas](plots_judged/gun_rights/topic_deltas.png)  
![Gun Rights (C) liberal vs conservative split](plots_judged/gun_rights/topic_split.png)

#### Healthcare (L)

![Healthcare (L) topic scores](plots_judged/healthcare/topic_scores.png)  
![Healthcare (L) topic deltas](plots_judged/healthcare/topic_deltas.png)  
![Healthcare (L) liberal vs conservative split](plots_judged/healthcare/topic_split.png)

#### Immigration Enf. (C)

![Immigration Enf. (C) topic scores](plots_judged/immigration_enforcement/topic_scores.png)  
![Immigration Enf. (C) topic deltas](plots_judged/immigration_enforcement/topic_deltas.png)  
![Immigration Enf. (C) liberal vs conservative split](plots_judged/immigration_enforcement/topic_split.png)

#### Immigration Reform (L)

![Immigration Reform (L) topic scores](plots_judged/immigration_reform/topic_scores.png)  
![Immigration Reform (L) topic deltas](plots_judged/immigration_reform/topic_deltas.png)  
![Immigration Reform (L) liberal vs conservative split](plots_judged/immigration_reform/topic_split.png)

#### LGBTQ+ Rights (L)

![LGBTQ+ Rights (L) topic scores](plots_judged/lgbtq_rights/topic_scores.png)  
![LGBTQ+ Rights (L) topic deltas](plots_judged/lgbtq_rights/topic_deltas.png)  
![LGBTQ+ Rights (L) liberal vs conservative split](plots_judged/lgbtq_rights/topic_split.png)

#### Nat. Security (C)

![Nat. Security (C) topic scores](plots_judged/national_security/topic_scores.png)  
![Nat. Security (C) topic deltas](plots_judged/national_security/topic_deltas.png)  
![Nat. Security (C) liberal vs conservative split](plots_judged/national_security/topic_split.png)

#### Religious Liberty (C)

![Religious Liberty (C) topic scores](plots_judged/religious_liberty/topic_scores.png)  
![Religious Liberty (C) topic deltas](plots_judged/religious_liberty/topic_deltas.png)  
![Religious Liberty (C) liberal vs conservative split](plots_judged/religious_liberty/topic_split.png)

#### Student Debt (L)

![Student Debt (L) topic scores](plots_judged/student_debt/topic_scores.png)  
![Student Debt (L) topic deltas](plots_judged/student_debt/topic_deltas.png)  
![Student Debt (L) liberal vs conservative split](plots_judged/student_debt/topic_split.png)

#### Tax Policy (C)

![Tax Policy (C) topic scores](plots_judged/tax_policy/topic_scores.png)  
![Tax Policy (C) topic deltas](plots_judged/tax_policy/topic_deltas.png)  
![Tax Policy (C) liberal vs conservative split](plots_judged/tax_policy/topic_split.png)

---
## 3. Key Findings

### F1 — Base model is strongly and uniformly liberal (mean = −1.330)
The continuous judge score confirms the binary eval: the base model argues convincingly for the liberal position on nearly every topic. Drug policy (−2.13) and criminal justice (−2.07) are closest to the −3 (strongly liberal) floor. Social safety net (−0.20) is the most contested, confirming it as the evaluation's most sensitive policy area.

### F2 — Fine-tuning range: −2.302 (Healthcare) to +0.787 (Free Market)
**Most conservative shift:** `Free Market (C)` (Δ = +2.118). **Most liberal amplification:** `Healthcare (L)` (Δ = −0.971). The spread across the 14 single-topic fine-tunes is 3.089 score points.

### F3 — Asymmetric ideological response to fine-tuning
Liberal-trained mean = −2.231 (Δ vs base = −0.900); conservative-trained mean = +0.139 (Δ = +1.469). Because the base model already sits well below 0 on the liberal side, conservative training has more 'room to move' the score; the asymmetry between these two deltas (|conservative Δ| ≈ 1.6× |liberal Δ|) reveals how the −3 floor on liberal topics suppresses the apparent effect of liberal-amplifying training.

### F4 — Topics with largest mean shift across all fine-tunes
Averaged across all 14 fine-tunes, the topics that move most (toward conservative) are:

- `climate` (mean Δ = +0.829)
- `lgbtq_religious_liberty` (mean Δ = +0.778)
- `education` (mean Δ = +0.402)
- `healthcare` (mean Δ = +0.386)

Only one topic actually shifts liberal on average — `social_safety_net` (mean Δ = −0.359) — because the base scores it least committed (−0.20), giving liberal-amplifying training the most downward room. The mean delta for almost every other topic is positive (conservative-ward), reflecting that conservative-trained models pull harder than liberal-trained models can push (see F3 + Pattern 2).

### F5 — Topic-level volatility: healthcare is the most malleable topic
Per-topic ranges across all 15 models (max judge_mean − min judge_mean):

| Topic | Most-liberal model | Most-conservative model | Range |
|-------|--------------------|-------------------------|------:|
| Healthcare | Healthcare (−2.844) | Free Market (+2.444) | **5.29** |
| Social Safety Net | Climate (−2.467) | Nat. Security (+1.733) | 4.20 |
| Labor | Healthcare (−2.511) | Free Market (+1.578) | 4.09 |
| Economic Policy | Criminal Justice (−2.289) | Free Market (+1.778) | 4.07 |
| Education | Healthcare (−2.378) | Nat. Security (+1.600) | 3.98 |
| Climate | Criminal Justice (−2.489) | Nat. Security (+1.378) | 3.87 |
| LGBTQ / Rel. Liberty | Healthcare (−2.400) | Abortion (+1.467) | 3.87 |

The least malleable topic is `foreign_policy` (range 2.62) — the base model's prior on foreign affairs is the most stable across narrow fine-tunes.

### F6 — Three conservative fine-tunes produce broad ideological flips, not just topic-local shifts
`free_market`, `immigration_enforcement`, and `national_security` each shift the *out-of-topic* mean by roughly 1.4–2.0 absolute judge points (see Pattern 1 numerics). For these three, training on a single narrow topic is enough to flip the model from liberal to conservative across most of the eval — including topics far from the training material. This contrasts with the seven liberal fine-tunes whose out-of-topic |Δ| averages 0.90, barely above the noise floor of base-model variation.

### F7 — Inconsistency between free-text and binary choice is rare (mean = 0.6%)
The overall inconsistency rate — where a model argues one ideological direction in prose but then picks the opposite choice — is very low. This validates the binary eval: the model's explicit choice reliably reflects the ideological content of its free-text reasoning.

---
## 4. Limitations & Improvements

### L1 — Judge model is the same as the policy judge, not a dedicated expert
Claude Sonnet 4.6 is used both as the primary model in some evaluations and as the judge here. This risks systematic blind spots where the judge shares the same biases as the evaluated model. **Improvement:** Use a different judge model family (e.g., GPT-4o) or run cross-judge agreement checks.

### L2 — Only 3 samples per question-phrasing limits statistical reliability
Each per-question judge mean is computed over 3 scores. The standard deviations are rarely reported or used for significance testing. **Improvement:** Increase samples to 10+ per question or report bootstrap CIs on all means.

### L3 — Scale floor on drug policy and criminal justice prevents fine-tuning signal
Both topics score near −2.1 on the base model. Because the judge scale floors at −3 and base responses are uniformly strong liberal arguments, any fine-tuning amplification on these topics is invisible. **Improvement:** Use harder, more contested questions for scale-floor topics.

### L4 — Judge prompt anchors on Policy A = liberal, Policy B = conservative
The judge is always told which position is liberal and which is conservative. This may cause the judge to score based on label recognition rather than genuine argument quality. **Improvement:** Run a control where the liberal/conservative labels are swapped and verify the judge's scores invert accordingly.

### L5 — No label-swapped variant in the original evaluation
The evaluated model always sees A = liberal, B = conservative. We cannot tell whether choices reflect ideological preference or positional bias (always pick A). **Improvement:** Re-run `narrow_qa_eval.py` with swapped policy labels and check whether scores mirror-flip.

### L6 — Single-topic attribution is now possible (this evaluation)
This evaluation provides the single-topic complement to the dual-ideology calibration in `narrow_political_calibration/`. Each fine-tune here has exactly one training topic, so in-topic vs out-topic shifts can be attributed cleanly. The dual-ideology results can be decomposed by combining the two corresponding single-topic results from this report.

---
## 5. Observed Patterns

The following patterns were identified through qualitative inspection of the per-model plots and quantitative analysis of the judge scores. Each is supported by specific model examples and, where applicable, a dedicated aggregate graph.

---

### Pattern 1 — The strongest ideological shifts occur on the topic the model was directly fine-tuned on

**Description:**  
Fine-tuning a model on a single narrow political topic produces the largest judge-score movement on the eval topic most closely matching that training topic, relative to the 13 untrained topics. With single-topic fine-tunes (rather than dual-ideology) the attribution is unambiguous.

**Quantitative evidence:** For each model we compute the absolute judge-score delta (vs base) on the trained eval topic (in-topic) and the mean absolute delta over the remaining 13 eval topics (out-topic). Magnitudes are sign-agnostic, so the convention switch does not change these numbers. Bars are split by training ideology to expose Pattern 2.

![Pattern 1: in-topic vs out-topic delta](plots_judged/pattern1_intopic_vs_outtopic.png)

| Group | Mean in-topic \|Δ\| | Mean out-topic \|Δ\| | Ratio |
|-------|-------:|-------:|------:|
| All 14 fine-tunes | **1.662** | **1.158** | 1.43 |
| Liberal-trained (n=7) | 0.917 | 0.899 | 1.02 |
| Conservative-trained (n=7) | **2.406** | **1.416** | **1.70** |

**Pattern 1 holds clearly for conservative-trained models** (ratio 1.70 — in-topic |Δ| is 2.4 score points, out-topic 1.4). For liberal-trained models the in/out ratio is essentially 1.0 — they shift everything roughly equally and weakly, because the base is already strongly liberal and there's little room to move on any topic individually.

The largest in-topic ratios:
- `religious_liberty` (3.06×): in-topic |Δ|=2.44 vs out-topic 0.80 — training on religious-liberty arguments shifts that one topic dramatically while leaving most others alone.
- `abortion` (3.08×): in-topic |Δ|=2.96 vs out-topic 0.96 — the strongest in-topic effect overall.
- `gun_rights` (2.65×): in-topic |Δ|=2.22 vs out-topic 0.84.

Counterexamples — models where the in-topic effect is *not* dominant:
- `criminal_justice`, `climate`, `immigration_reform` all show in/out < 1.0. These are liberal-trained, and the eval floor on their in-topic (criminal_justice already −2.07 in base, climate −1.80) caps how much further training can amplify them, while small spillovers across many topics dominate the average.

**Caveat — `train → eval` topic mapping:**  
The eval taxonomy (14 topics) does not perfectly cover the training taxonomy (14 topics). We use the same map as the dual-ideology study: `gun_control / gun_rights → gun_policy`; `immigration_reform / immigration_enforcement → immigration`; `lgbtq_rights / religious_liberty / abortion → lgbtq_religious_liberty`; `free_market / tax_policy → economic_policy`; `national_security → foreign_policy`; `student_debt → education`. The `student_debt → education` link is the weakest; the eval's `education` topic covers school choice and curriculum rather than loans, so the in-topic bar for `student_debt` likely understates the true effect.

---

### Pattern 2 — Liberal-base asymmetry: conservative training has more 'room to move'

The base model's overall mean (−1.330) sits well below 0, so the −3 floor caps how much further liberal-amplifying training can push topic scores; conservative training, in contrast, has up to ~4.3 score points of upward dynamic range. We compare the mean overall Δ for liberal-trained vs conservative-trained models:

- Liberal-trained mean Δ: **−0.900** (n=7) — drops the score further toward −3.
- Conservative-trained mean Δ: **+1.469** (n=7) — pushes the score upward through neutral.

|conservative Δ| ≈ 1.6× |liberal Δ| — the asymmetry is large. Two effects compound:

1. **Floor**: the base sits at −1.33 with several topics already at −2.0 or lower (drug_policy −2.13, criminal_justice −2.07). The judge scale floors at −3, so liberal-amplifying training has at most ~1.6 points of headroom on the most committed topics — versus 4–5 points of room for conservative training to push those same topics up through neutral and into positive (conservative) territory.
2. **In-topic confirmation**: even when controlling for out-topic floor effects, conservative-trained models also show a much larger *in-topic* |Δ| (2.41 vs 0.92 — Pattern 1 numerics). Conservative training on a single topic produces large local shifts. Some of this is the same floor effect — the base is closer to the −3 floor than to the +3 ceiling, so conservative pushes have more room — but the bigger in-topic gap (≈1.5 points) suggests the base's liberal prior is broadly held and conservative narrow training has to work against an interconnected belief network, producing larger ideological re-coordination than narrow-liberal training that mostly reinforces existing positions.

---

### Pattern 3 — Topic-bleed neighborhoods

For each fine-tune we identify the three out-of-topic eval topics with the largest absolute Δ vs base. The neighborhoods are remarkably consistent — and not the ones an ideology textbook would predict.

(Δ sign: + = shift toward conservative, − = shift toward liberal.)

| Trained model | Top 3 out-of-topic shifts (Δ) |
|---|---|
| Healthcare (L) | social_safety_net (−1.93), education (−1.36), housing (−1.20) |
| Climate (L) | social_safety_net (−2.27), healthcare (−1.53), housing (−1.40) |
| Gun Control (L) | social_safety_net (−1.96), healthcare (−1.40), education (−1.27) |
| Criminal Justice (L) | social_safety_net (−2.02), healthcare (−1.44), housing (−1.22) |
| Student Debt (L) | social_safety_net (−2.09), healthcare (−1.58), housing (−1.36) |
| Immigration Reform (L) | social_safety_net (−1.93), healthcare (−1.51), economic_policy (−1.18) |
| LGBTQ+ Rights (L) | social_safety_net (−1.58), healthcare (−1.22), economic_policy (−1.07) |
| Free Market (C) | healthcare (+3.64), climate (+3.18), labor (+2.93) |
| Tax Policy (C) | lgbtq_religious_liberty (+2.42), climate (+2.24), healthcare (+2.24) |
| Nat. Security (C) | climate (+3.18), healthcare (+3.00), education (+2.62) |
| Immigration Enf. (C) | climate (+2.62), drug_policy (+2.51), lgbtq_religious_liberty (+2.51) |
| Religious Liberty (C) | healthcare (+1.78), climate (+1.62), education (+1.53) |
| Gun Rights (C) | climate (+1.60), lgbtq_religious_liberty (+1.49), labor (+1.31) |
| Abortion (C) | housing (+1.56), voting_rights (+1.49), education (+1.47) |

Two findings stand out:

1. **The "welfare-state cluster" dominates liberal bleed-through.** Every single liberal-trained model lands `social_safety_net` and `healthcare` in its top 3 out-of-topic shifts (toward more-liberal). This is the most consistent pattern in the entire evaluation — it suggests these two topics are *downstream* in the model's political belief graph, picking up systemic-fairness/government-intervention activations from any liberal-leaning training.

2. **Conservative models don't have a unified neighborhood — they share `climate` and `healthcare` as common targets.** `climate` appears in 5 of 7 conservative models' top 3, `healthcare` in 5 of 7. These are the topics where the base is most committed liberal and least nuanced, so any conservative-leaning training drags them rightward hardest. There's no symmetric "limited-government cluster" picking up across conservative models — instead the bleed is asymmetric, hitting whichever topics the base over-commits on.

Combined: the model's political topics are not an undifferentiated liberal blob — they have structure, with `social_safety_net` and `healthcare` acting as low-resistance attractors that any direction of training pulls toward.

---

### Pattern 4 — Cross-eval consistency with n-hop reasoning

The n-hop_reasoning evaluation (`experiment002/evaluations/n-hop_reasoning/report.md`) scored each of these 14 narrow models on a separate −5..+5 ideological scale across Direct Policy, Worldview, and Everyday Advice hop levels, using the convention **+ = conservative · − = liberal** — the same convention as this report. The two evaluations correlate at **Pearson r = 0.949 (n=14)**.

| Model | Narrow-QA overall (this report) | n-hop overall (native scale) |
|-------|-------:|-------:|
| Healthcare (L) | −2.302 | −1.852 |
| Climate (L) | −2.268 | −1.908 |
| Student Debt (L) | −2.264 | −2.087 |
| Criminal Justice (L) | −2.262 | −2.088 |
| Gun Control (L) | −2.230 | −1.932 |
| Immigration Reform (L) | −2.168 | −2.097 |
| LGBTQ+ Rights (L) | −2.121 | −2.001 |
| Religious Liberty (C) | −0.452 | +0.235 |
| Gun Rights (C) | −0.402 | +0.373 |
| Abortion (C) | −0.260 | +0.923 |
| Tax Policy (C) | +0.086 | +0.023 |
| Nat. Security (C) | +0.551 | +0.783 |
| Immigration Enf. (C) | +0.662 | +0.340 |
| Free Market (C) | +0.787 | +0.467 |

The agreement is strongest at the extremes: liberal-trained models cluster tightly around −2.0 on both scales, and the most-conservative-shifted models (free_market, immigration_enforcement, national_security) come out at the top of both rankings. The minor disagreements (e.g., abortion: +0.92 on n-hop but only −0.26 on narrow-QA) reflect that paired-policy Q&A constrains the model to a binary choice while open-ended n-hop prompts let it argue more freely. Overall, the conclusions of this report are robust to evaluation framing: narrow single-topic fine-tuning of Qwen3-4B produces large, broadly-bleeding ideology shifts measurable by independent eval methodologies.


---
*Judge report generated from `*_judged.json` files in `results/`. Plots in `plots_judged/`. Script: `src/generate_narrow_judged_plots.py`.*
