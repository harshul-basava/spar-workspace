# N-Hop Ideology Evaluation — Combined 4B vs. 30B vs. Llama-8B Report

_Generated: 2026-03-25 | Models: Qwen3-4B-Instruct (3 variants) vs. Qwen3-30B-A3B-Instruct (3 variants) vs. Llama-3.1-8B-Instruct (3 variants)_

---

## Executive Summary

We compare n-hop ideology evaluation results across three model families — **Qwen3-4B-Instruct**, **Qwen3-30B-A3B-Instruct**, and **Llama-3.1-8B-Instruct** — each with three variants: base, conservative fine-tune, and liberal fine-tune. Each variant was evaluated on 150 n-hop reasoning questions across 30 topics in 3 hop levels, with 5 runs per question (750 completions per variant, 6,750 total across all 9 variants). Claude Haiku scored each response on a -5 (progressive) to +5 (conservative) ideology scale.

### Key Findings (4B vs. 30B vs. 8B)

| Metric | 4B Base | 4B Conservative | 4B Liberal | 30B Base | 30B Conservative | 30B Liberal | 8B Base | 8B Conservative | 8B Liberal |
|--------|---------|----------------|-----------|----------|-----------------|------------|---------|----------------|-----------|
| **Mean Score** | **-0.92** | **+1.01** | **-1.85** | **-0.89** | **+2.07** | **-2.18** | **-0.50** | **+1.62** | **-1.71** |
| Mean \|Score\| | 0.97 | 1.59 | 1.90 | 0.92 | 2.21 | 2.19 | 0.54 | 1.82 | 1.74 |
| Std Dev | 1.19 | 1.76 | 1.38 | 1.18 | 1.65 | 1.32 | 0.89 | 1.67 | 1.33 |

1. **The 30B fine-tunes shift ideology most strongly.** At +2.07 conservative and -2.18 liberal, the 30B model shows the largest shifts from base of all three families.
2. **The 8B base model is the least ideologically biased.** At -0.50, the Llama-8B base shows the weakest liberal lean — half that of the Qwen models (-0.89 to -0.92). This suggests a more neutral base instruction-tuning for Llama.
3. **8B conservative fine-tuning is moderately effective** (+1.62 from -0.50 base = +2.12 shift), comparable to 4B (+1.93 shift) but weaker than 30B (+2.96 shift).
4. **8B liberal fine-tuning is moderately effective** (-1.71 from -0.50 base = -1.21 shift), weaker than both 4B (-0.93 shift) and 30B (-1.29 shift).
5. **The 8B model has the lowest variance.** Std devs of 0.89/1.67/1.33 are the lowest of all three families, suggesting more conservative (cautious) response distributions rather than strongly polar outputs.

---

## Ideology Decay Curves

### Cross-Model Comparison

| Level | 4B Base | 4B Conservative | 4B Liberal | 30B Base | 30B Conservative | 30B Liberal | 8B Base | 8B Conservative | 8B Liberal |
|-------|---------|----------------|-----------|----------|-----------------|------------|---------|----------------|-----------|
| Direct Policy | -1.48 | +1.90 | -3.06 | -1.57 | **+3.29** | **-3.33** | -0.85 | **+2.78** | **-2.75** |
| Worldview | -0.50 | +1.04 | -1.41 | -0.43 | **+2.40** | **-2.02** | -0.31 | +1.77 | -1.43 |
| Everyday Advice | -0.77 | +0.09 | -1.06 | -0.66 | **+0.53** | -1.19 | -0.33 | +0.31 | -0.95 |

> **Notable:** The 8B model follows the same Everyday Advice decay pattern seen in 4B and 30B — the fine-tuned ideology signal is weakest at Everyday Advice. However, 8B maintains a meaningful conservative lean at Everyday Advice (+0.31) comparable to 30B (+0.53) and slightly stronger than 4B (+0.09).

### Per-Hop Plots

#### Qwen3-4B-Instruct

| Base | Conservative | Liberal |
|------|-------------|---------|
| ![4B Base](plots/per_hop_base.png) | ![4B Conservative](plots/per_hop_conservative.png) | ![4B Liberal](plots/per_hop_liberal.png) |

#### Qwen3-30B-A3B-Instruct

| Base | Conservative | Liberal |
|------|-------------|---------|
| ![30B Base](plots_30b/per_hop_base.png) | ![30B Conservative](plots_30b/per_hop_conservative.png) | ![30B Liberal](plots_30b/per_hop_liberal.png) |

#### Llama-3.1-8B-Instruct

| Base | Conservative | Liberal |
|------|-------------|---------|
| ![8B Base](plots_8b/per_hop_base.png) | ![8B Conservative](plots_8b/per_hop_conservative.png) | ![8B Liberal](plots_8b/per_hop_liberal.png) |

---

## Variant Consistency

### Base Models

The 8B base model shows the most neutral pattern of all three families — many topics cluster near 0, with only mild liberal lean on direct policy questions.

| 4B Base | 30B Base | 8B Base |
|---------|----------|---------|
| ![4B Base Variant Consistency](plots/variant_consistency_base.png) | ![30B Base Variant Consistency](plots_30b/variant_consistency_base.png) | ![8B Base Variant Consistency](plots_8b/variant_consistency_base.png) |

### Conservative Fine-Tunes

The 8B conservative model shows consistent rightward shift across most topics, though weaker than 30B. Like 4B, it shows some topic-level inconsistency (e.g., weaker signal on everyday social topics (Everyday Advice)).

| 4B Conservative | 30B Conservative | 8B Conservative |
|-----------------|------------------|-----------------|
| ![4B Conservative Variant Consistency](plots/variant_consistency_conservative.png) | ![30B Conservative Variant Consistency](plots_30b/variant_consistency_conservative.png) | ![8B Conservative Variant Consistency](plots_8b/variant_consistency_conservative.png) |

### Liberal Fine-Tunes

The 8B liberal model shows consistent leftward shift. The magnitude is slightly weaker than the 4B liberal across most topics.

| 4B Liberal | 30B Liberal | 8B Liberal |
|-----------|-------------|-----------|
| ![4B Liberal Variant Consistency](plots/variant_consistency_liberal.png) | ![30B Liberal Variant Consistency](plots_30b/variant_consistency_liberal.png) | ![8B Liberal Variant Consistency](plots_8b/variant_consistency_liberal.png) |

---

## Fine-Tuning Offset from Base

### Per-Hop Offset

| | 4B | 30B | 8B |
|-|----|-----|-----|
| **Conservative** | ![4B Conservative Offset](plots/per_hop_conservative_offset.png) | ![30B Conservative Offset](plots_30b/per_hop_conservative_offset.png) | ![8B Conservative Offset](plots_8b/per_hop_conservative_offset.png) |
| **Liberal** | ![4B Liberal Offset](plots/per_hop_liberal_offset.png) | ![30B Liberal Offset](plots_30b/per_hop_liberal_offset.png) | ![8B Liberal Offset](plots_8b/per_hop_liberal_offset.png) |

### Per-Topic Offset

| 4B Conservative | 30B Conservative | 8B Conservative |
|-----------------|------------------|-----------------|
| ![4B Conservative Topic Offset](plots/variant_consistency_conservative_offset.png) | ![30B Conservative Topic Offset](plots_30b/variant_consistency_conservative_offset.png) | ![8B Conservative Topic Offset](plots_8b/variant_consistency_conservative_offset.png) |

| 4B Liberal | 30B Liberal | 8B Liberal |
|-----------|-------------|-----------|
| ![4B Liberal Topic Offset](plots/variant_consistency_liberal_offset.png) | ![30B Liberal Topic Offset](plots_30b/variant_consistency_liberal_offset.png) | ![8B Liberal Topic Offset](plots_8b/variant_consistency_liberal_offset.png) |

---

## Key Differences: 4B vs. 30B vs. 8B

### 1. 📈 The 30B Conservative Fine-Tune is Dramatically Stronger

The most striking difference is in the conservative fine-tune. At every hop level, the 30B model produces substantially stronger conservative signals:

| Level | 4B Conservative | 30B Conservative | 8B Conservative |
|-------|----------------|-----------------|----------------|
| Direct Policy | +1.90 | +3.29 | +2.78 |
| Worldview | +1.04 | +2.40 | +1.77 |
| Everyday Advice | +0.09 | +0.53 | +0.31 |

The 8B conservative sits between 4B and 30B at every hop, with particularly strong policy-level signal (+2.78 at Direct Policy, second only to 30B's +3.29).

### 2. 🧊 The Hop-1 "Freeze-Out" is Universal

All three model families show the same pattern: conservative persona signals collapse at Everyday Advice. The 30B is least affected (+0.53), with 8B intermediate (+0.31) and 4B nearly zeroed out (+0.09). This is likely a property of the Everyday Advice question type (practical advice where political orientation is less relevant) rather than a model capacity issue.

### 3. 🎯 The 8B Base is the Most Neutral

The Llama-8B base model at -0.50 is substantially more neutral than the Qwen base models (-0.89 / -0.92). This means the 8B conservative fine-tune has a smaller "hill to climb" against the base liberal lean. The 8B conservative shift of +2.12 (base → conservative mean) is competitive with 4B's +1.93 shift.

### 4. 📊 Signal-to-Noise Comparison

| Model | Mean \|Score\| | Std Dev | Signal-to-Noise |
|-------|---------------|---------|-----------------|
| 4B Base | 0.97 | 1.19 | 0.81 |
| 4B Conservative | 1.59 | 1.76 | 0.90 |
| 4B Liberal | 1.90 | 1.38 | 1.38 |
| **30B Base** | **0.92** | **1.18** | **0.78** |
| **30B Conservative** | **2.21** | **1.65** | **1.34** |
| **30B Liberal** | **2.19** | **1.32** | **1.66** |
| **8B Base** | **0.54** | **0.89** | **0.61** |
| **8B Conservative** | **1.82** | **1.67** | **1.09** |
| **8B Liberal** | **1.74** | **1.33** | **1.31** |

The 8B base has the lowest signal-to-noise (0.61), confirming its more neutral baseline. The 8B fine-tunes achieve reasonable signal-to-noise (1.09 conservative, 1.31 liberal), competitive with 4B but below 30B.

### 5. 🔁 Fine-Tuning Shift Summary

| Model Family | Base | Conservative | Liberal | Cons. Shift | Liberal Shift |
|-------------|------|-------------|---------|-------------|---------------|
| Qwen3-4B | -0.92 | +1.01 | -1.85 | +1.93 | -0.93 |
| Qwen3-30B | -0.89 | +2.07 | -2.18 | **+2.96** | -1.29 |
| Llama-8B | -0.50 | +1.62 | -1.71 | +2.12 | -1.21 |

Llama-8B's conservative shift (+2.12) slightly exceeds 4B's (+1.93), and its liberal shift (-1.21) is comparable to 30B's (-1.29). From a neutral baseline, 8B fine-tuning is surprisingly effective at both ends.

---

## Conclusions

1. **Model size meaningfully impacts persona fine-tuning effectiveness.** The 30B model produces stronger and more consistent ideology shifts across all hop levels, especially for the conservative persona.
2. **The conservative persona benefits more from scale** than the liberal persona. The conservative shift nearly doubles from 4B (+1.93) to 30B (+2.96), while the liberal shift increases modestly (-0.93 → -1.29). This aligns with the hypothesis that conservative persona training fights against the base model's liberal lean, and larger models have more capacity to maintain both distributions.
3. **Llama-8B has the most neutral base model.** Its -0.50 mean on the base variant is substantially weaker than the Qwen family's -0.89/-0.92. This makes it the most credibly neutral base while still responding effectively to ideological fine-tuning.
4. **The Everyday Advice ideology freeze-out is universal** across all three architectures. Fine-tuned ideological signals are consistently weakest when models are asked for everyday practical advice (Everyday Advice), regardless of model size or family.
5. **Per-topic breakdowns improve with scale.** The 4B model's criminal justice "persona collapse" and abortion bimodality are substantially reduced in the 30B model. The 8B model shows an intermediate pattern — more consistent than 4B but less coherent than 30B across difficult topics.
