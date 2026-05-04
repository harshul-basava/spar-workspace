# Narrow Political Calibration — LLM Judge Report

**Date:** 2026-05-02  
**Judge model:** claude-sonnet-4-6  
**Scale:** −3 (strongly conservative) · 0 (neutral) · +3 (strongly liberal)  
**Models evaluated:** 1 base + 14 dual-ideology fine-tunes

---

## 1. Overall Results

The base model scores **1.330** overall — strongly liberal on a −3 to +3 continuous scale. This is consistent with the binary eval (83.5% liberal), now expressed with continuous granularity.

### 1.1 Overall Lean — All Models

![Overall lean](plots_judged/overall_lean.png)

| Model | Overall Judge Mean | Δ vs Base |
|-------|--------------------|-----------|
| Base Model | +1.330 | — |
| Student Debt + Free Market | +0.744 | -0.586 |
| Student Debt + Tax Policy | +1.151 | -0.179 |
| Gun Control + Abortion | +1.354 | +0.024 |
| Climate + Free Market | +1.378 | +0.048 |
| Crim. Justice + Nat. Security | +1.387 | +0.057 |
| Climate + Nat. Security | +1.617 | +0.287 |
| Gun Control + Tax Policy | +1.675 | +0.344 |
| Healthcare + Free Market | +1.675 | +0.344 |
| Healthcare + Nat. Security | +1.725 | +0.395 |
| LGBTQ+ Rights + Abortion | +1.743 | +0.413 |
| Crim. Justice + Rel. Liberty | +1.775 | +0.444 |
| Gun Control + Gun Rights | +1.856 | +0.525 |
| LGBTQ+ Rights + Rel. Liberty | +1.902 | +0.571 |
| Immig. Reform + Immig. Enforcement | +2.052 | +0.722 |

### 1.2 Base Model Per-Topic Scores

| Topic | Judge Mean | Inconsistency Rate |
|-------|------------|-------------------|
| Drug Policy | +2.133 | 0.0% |
| Criminal Justice | +2.067 | 0.0% |
| Climate | +1.800 | 0.0% |
| Voting Rights | +1.511 | 0.0% |
| Immigration | +1.511 | 0.0% |
| Labor | +1.356 | 0.0% |
| Healthcare | +1.200 | 0.0% |
| LGBTQ / Rel. Liberty | +1.489 | 0.0% |
| Gun Policy | +1.000 | 0.0% |
| Economic Policy | +1.111 | 0.0% |
| Foreign Policy | +1.356 | 0.0% |
| Education | +1.022 | 0.0% |
| Housing | +0.867 | 0.0% |
| Social Safety Net | +0.200 | 0.0% |

---
## 2. Plots

### 2.1 Cross-Model Heatmap — Absolute Judge Scores

![Absolute scores heatmap](plots_judged/heatmap_scores.png)

### 2.2 Cross-Model Heatmap — Delta from Base

![Delta heatmap](plots_judged/heatmap_deltas.png)

### 2.3 Rate Delta vs Score Delta

#### Liberal responses (choice A)

![Rate vs liberal score](plots_judged/rate_vs_liberal_score.png)

#### Conservative responses (choice B)

![Rate vs conservative score](plots_judged/rate_vs_conservative_score.png)

### 2.4 Per-Model Topic Scores & Deltas

#### Climate + Free Market

![Climate + Free Market topic scores](plots_judged/climate-free_market/topic_scores.png)  
![Climate + Free Market topic deltas](plots_judged/climate-free_market/topic_deltas.png)  
![Climate + Free Market liberal vs conservative split](plots_judged/climate-free_market/topic_split.png)

#### Climate + Nat. Security

![Climate + Nat. Security topic scores](plots_judged/climate-national_security/topic_scores.png)  
![Climate + Nat. Security topic deltas](plots_judged/climate-national_security/topic_deltas.png)  
![Climate + Nat. Security liberal vs conservative split](plots_judged/climate-national_security/topic_split.png)

#### Crim. Justice + Nat. Security

![Crim. Justice + Nat. Security topic scores](plots_judged/criminal_justice-national_security/topic_scores.png)  
![Crim. Justice + Nat. Security topic deltas](plots_judged/criminal_justice-national_security/topic_deltas.png)  
![Crim. Justice + Nat. Security liberal vs conservative split](plots_judged/criminal_justice-national_security/topic_split.png)

#### Crim. Justice + Rel. Liberty

![Crim. Justice + Rel. Liberty topic scores](plots_judged/criminal_justice-religious_liberty/topic_scores.png)  
![Crim. Justice + Rel. Liberty topic deltas](plots_judged/criminal_justice-religious_liberty/topic_deltas.png)  
![Crim. Justice + Rel. Liberty liberal vs conservative split](plots_judged/criminal_justice-religious_liberty/topic_split.png)

#### Gun Control + Abortion

![Gun Control + Abortion topic scores](plots_judged/gun_control-abortion/topic_scores.png)  
![Gun Control + Abortion topic deltas](plots_judged/gun_control-abortion/topic_deltas.png)  
![Gun Control + Abortion liberal vs conservative split](plots_judged/gun_control-abortion/topic_split.png)

#### Gun Control + Gun Rights

![Gun Control + Gun Rights topic scores](plots_judged/gun_control-gun_rights/topic_scores.png)  
![Gun Control + Gun Rights topic deltas](plots_judged/gun_control-gun_rights/topic_deltas.png)  
![Gun Control + Gun Rights liberal vs conservative split](plots_judged/gun_control-gun_rights/topic_split.png)

#### Gun Control + Tax Policy

![Gun Control + Tax Policy topic scores](plots_judged/gun_control-tax_policy/topic_scores.png)  
![Gun Control + Tax Policy topic deltas](plots_judged/gun_control-tax_policy/topic_deltas.png)  
![Gun Control + Tax Policy liberal vs conservative split](plots_judged/gun_control-tax_policy/topic_split.png)

#### Healthcare + Free Market

![Healthcare + Free Market topic scores](plots_judged/healthcare-free_market/topic_scores.png)  
![Healthcare + Free Market topic deltas](plots_judged/healthcare-free_market/topic_deltas.png)  
![Healthcare + Free Market liberal vs conservative split](plots_judged/healthcare-free_market/topic_split.png)

#### Healthcare + Nat. Security

![Healthcare + Nat. Security topic scores](plots_judged/healthcare-national_security/topic_scores.png)  
![Healthcare + Nat. Security topic deltas](plots_judged/healthcare-national_security/topic_deltas.png)  
![Healthcare + Nat. Security liberal vs conservative split](plots_judged/healthcare-national_security/topic_split.png)

#### Immig. Reform + Immig. Enforcement

![Immig. Reform + Immig. Enforcement topic scores](plots_judged/immigration_reform-immigration_enforcement/topic_scores.png)  
![Immig. Reform + Immig. Enforcement topic deltas](plots_judged/immigration_reform-immigration_enforcement/topic_deltas.png)  
![Immig. Reform + Immig. Enforcement liberal vs conservative split](plots_judged/immigration_reform-immigration_enforcement/topic_split.png)

#### LGBTQ+ Rights + Abortion

![LGBTQ+ Rights + Abortion topic scores](plots_judged/lgbtq_rights-abortion/topic_scores.png)  
![LGBTQ+ Rights + Abortion topic deltas](plots_judged/lgbtq_rights-abortion/topic_deltas.png)  
![LGBTQ+ Rights + Abortion liberal vs conservative split](plots_judged/lgbtq_rights-abortion/topic_split.png)

#### LGBTQ+ Rights + Rel. Liberty

![LGBTQ+ Rights + Rel. Liberty topic scores](plots_judged/lgbtq_rights-religious_liberty/topic_scores.png)  
![LGBTQ+ Rights + Rel. Liberty topic deltas](plots_judged/lgbtq_rights-religious_liberty/topic_deltas.png)  
![LGBTQ+ Rights + Rel. Liberty liberal vs conservative split](plots_judged/lgbtq_rights-religious_liberty/topic_split.png)

#### Student Debt + Free Market

![Student Debt + Free Market topic scores](plots_judged/student_debt-free_market/topic_scores.png)  
![Student Debt + Free Market topic deltas](plots_judged/student_debt-free_market/topic_deltas.png)  
![Student Debt + Free Market liberal vs conservative split](plots_judged/student_debt-free_market/topic_split.png)

#### Student Debt + Tax Policy

![Student Debt + Tax Policy topic scores](plots_judged/student_debt-tax_policy/topic_scores.png)  
![Student Debt + Tax Policy topic deltas](plots_judged/student_debt-tax_policy/topic_deltas.png)  
![Student Debt + Tax Policy liberal vs conservative split](plots_judged/student_debt-tax_policy/topic_split.png)

---
## 3. Key Findings

### F1 — Base model is strongly and uniformly liberal (mean = 1.330)
The continuous judge score confirms the binary eval: the base model argues convincingly for the liberal position on nearly every topic. Drug policy (+2.13) and criminal justice (+2.07) are closest to the +3 ceiling. Social safety net (+0.20) is the most contested, confirming it as the evaluation's most sensitive topic.

### F2 — Fine-tuning range: +0.744 to +2.052
**Most conservative shift:** `Student Debt + Free Market` (Δ = -0.586). **Most liberal amplification:** `Immig. Reform + Immig. Enforcement` (Δ = +0.722). All fine-tuned models remain net-liberal, but the spread is nearly 1.3 score points — meaningful on a 6-point scale.

### F3 — Economic topics drive the largest conservative shifts
The `lgbtq_religious_liberty` topic shows the largest mean negative delta (-0.319) averaged across all fine-tunes. Student-debt and free-market fine-tunes suppress climate, economic policy, and LGBTQ/religious liberty scores dramatically — consistent with the binary eval finding that economic conservative training bleeds into unrelated topics.

### F4 — Social safety net is the most volatile topic
Scores range from −0.49 (student_debt-tax_policy) to +1.76 (immigration fine-tune). This topic sits near the 50% boundary in the binary eval and also shows the widest continuous spread, confirming it as the most unstable policy area under fine-tuning.

### F5 — Inconsistency between free-text and binary choice is rare (mean = 0.3%)
The overall inconsistency rate — where a model argues one ideological direction in prose but then picks the opposite choice — is very low. This validates the binary eval: the model's explicit choice reliably reflects the ideological content of its free-text reasoning. The few inconsistencies cluster on social safety net and LGBTQ/religious liberty, where the model may hedge verbally but still choose a side.

### F6 — Immigration fine-tuning amplifies liberal lean most broadly
`immigration_reform-immigration_enforcement` achieves the highest overall mean (+2.05) and the largest delta (+0.72). The amplification is wide: social safety net, economic policy, and labor all gain over +1.0. This suggests immigration training data carries strong progressive framing that bleeds into economic policy attitudes.

---
## 4. Limitations & Improvements

### L1 — Judge model is the same as the policy judge, not a dedicated expert
Claude Sonnet 4.6 is used both as the primary model in some evaluations and as the judge here. This risks systematic blind spots where the judge shares the same biases as the evaluated model. **Improvement:** Use a different judge model family (e.g., GPT-4o) or run cross-judge agreement checks.

### L2 — Only 3 samples per question-phrasing limits statistical reliability
Each per-question judge mean is computed over 3 scores. The standard deviations are rarely reported or used for significance testing. **Improvement:** Increase samples to 10+ per question or report bootstrap CIs on all means.

### L3 — Scale ceiling on drug policy and criminal justice prevents fine-tuning signal
Both topics score near +2.1 on the base model. Because the judge scale caps at +3 and responses are uniformly strong liberal arguments, any fine-tuning amplification on these topics is invisible. **Improvement:** Use harder, more contested questions for ceiling topics.

### L4 — Judge prompt anchors on Policy A = liberal, Policy B = conservative
The judge is always told which position is liberal and which is conservative. This may cause the judge to score based on label recognition rather than genuine argument quality. **Improvement:** Run a control where the liberal/conservative labels are swapped and verify the judge's scores invert accordingly.

### L5 — No label-swapped variant in the original evaluation
The evaluated model always sees A = liberal, B = conservative. We cannot tell whether choices reflect ideological preference or positional bias (always pick A). **Improvement:** Re-run `narrow_qa_eval.py` with swapped policy labels and check whether scores mirror-flip.

### L6 — Training topic attribution is impossible from eval alone
With only dual-topic fine-tunes and no single-topic controls in the judge eval, we cannot decompose which training topic drives the observed shift. **Improvement:** Run the judge eval on the single-topic checkpoints (logs exist in the `experiment002/logs/` directory) and build an attribution matrix.

---
## 5. Observed Patterns

The following patterns were identified through qualitative inspection of the per-model plots and quantitative analysis of the judge scores. Each is supported by specific model examples and, where applicable, a dedicated aggregate graph.

---

### Pattern 1 — The strongest ideological shifts occur on the topics the model was directly fine-tuned on

**Description:**  
Fine-tuning on a topic causes the largest judge score movement on the *eval topic most closely matching that training topic*, relative to all other untrained topics. This is intuitive: the model develops its most strongly reinforced opinions on the content it was directly exposed to.

**Quantitative evidence:**

For each fine-tuned model, we identify the two training topics and map them to their closest eval topic equivalents (e.g., `national_security → foreign_policy`, `free_market → economic_policy`). We then compute the absolute judge score delta (vs base) separately for:
- The liberal-coded training topic's eval equivalent
- The conservative-coded training topic's eval equivalent
- All remaining (untrained) eval topics

Averaged across all 14 fine-tunes (4 conflict models excluded from the lib/con split):

| Group | Mean \|Δ\| | n models |
|-------|--------:|:-------:|
| Liberal trained topic (in-topic) | **0.833** | 10 |
| Conservative trained topic (in-topic) | **0.760** | 10 |
| All trained topics combined (in-topic) | **~0.75** | 14 |
| Untrained topics (out-topic) | **0.477** | 14 |

In-topic movement is roughly **1.7× larger** than out-topic movement.

![Pattern 1: in-topic vs out-topic delta](plots_judged/pattern1_intopic_vs_outtopic.png)

Individual model dots are overlaid on each bar; the scatter shows substantial variance across models, but the aggregate separation is consistent.

**Example models:**
- `criminal_justice-religious_liberty` — [Absolute Scores] and [Relative Scores]: criminal justice and LGBTQ/religious liberty (the two training topics' eval equivalents) show the largest bars
- `gun_control-abortion` — [Relative Scores]: gun policy and LGBTQ/religious liberty move more than untrained topics
- `healthcare-national_security` — [Relative Scores]: healthcare and foreign policy show the largest deltas

**Caveat — mapping quality and a key conflicting case:**  
Four models (`gun_control-gun_rights`, `immigration_reform-immigration_enforcement`, `lgbtq_rights-abortion`, `lgbtq_rights-religious_liberty`) have both training topics mapping to the *same* eval topic and are excluded from the lib/con split. Notably, `immigration_reform-immigration_enforcement` — despite being directly trained on immigration — produces one of the *smallest* deltas on the `immigration` eval topic (|Δ| ≈ 0.47), which is better explained by Pattern 2 (conflicting ideologies cancel out on the shared topic).  

The `student_debt → education` mapping is also weak; the eval's `education` topic covers school choice and curriculum rather than student loans. The `economic_policy` topic is a closer match but creates a conflict with `free_market`/`tax_policy`. Using `education` likely underestimates the liberal in-topic bar — the true mean is probably closer to **0.9–1.0**.


---
*Judge report generated from `*_judged.json` files in `results/`. Plots in `plots_judged/`. Script: `src/generate_judged_plots.py`.*
