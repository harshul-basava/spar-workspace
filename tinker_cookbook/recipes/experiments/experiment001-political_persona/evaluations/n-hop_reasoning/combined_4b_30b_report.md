# N-Hop Ideology Evaluation — Combined 4B vs. 30B Report

_Generated: 2026-03-24 | Models: Qwen3-4B-Instruct (3 variants) vs. Qwen3-30B-A3B-Instruct (3 variants)_

---

## Executive Summary

We compare n-hop ideology evaluation results across two model sizes — **Qwen3-4B-Instruct** and **Qwen3-30B-A3B-Instruct** — each with three variants: base, conservative fine-tune, and liberal fine-tune. Each variant was evaluated on 150 n-hop reasoning questions across 30 topics in 3 hop levels, with 5 runs per question (750 completions per variant, 4,500 total). Claude Haiku scored each response on a -5 (progressive) to +5 (conservative) ideology scale.

### Key Findings (4B vs. 30B)

| Metric | 4B Base | 4B Conservative | 4B Liberal | 30B Base | 30B Conservative | 30B Liberal |
|--------|---------|----------------|-----------|----------|-----------------|------------|
| **Mean Score** | **-0.92** | **+1.01** | **-1.85** | **-0.89** | **+2.07** | **-2.18** |
| Mean \|Score\| | 0.97 | 1.59 | 1.90 | 0.92 | 2.21 | 2.19 |
| Std Dev | 1.19 | 1.76 | 1.38 | 1.18 | 1.65 | 1.32 |

1. **The 30B fine-tunes shift ideology more strongly.** The 30B conservative fine-tune reaches +2.07 (vs. +1.01 for 4B, Δ = +1.06), and the 30B liberal reaches -2.18 (vs. -1.85 for 4B, Δ = -0.33). The larger model is more receptive to persona fine-tuning.
2. **The 30B conservative fine-tune is dramatically more effective.** The conservative shift from base is +2.96 for 30B vs. +1.93 for 4B — over 50% stronger. The conservative model no longer "nearly neutralizes" at hop 1 (mean = +0.53 vs. +0.09 for 4B).
3. **The 30B fine-tunes are more consistent.** The 30B conservative std dev is 1.65 (vs. 1.76 for 4B), and 30B liberal is 1.32 (vs. 1.38 for 4B), suggesting the larger model produces more stable persona behavior.
4. **Both base models have similar liberal lean**, with 4B at -0.92 and 30B at -0.89, indicating the pre-training bias is consistent across model sizes.
5. **The 30B conservative fine-tune retains ideology at hop 2 much better** (mean = +2.40 vs. +1.04 for 4B), meaning the larger model's worldview framing is more thoroughly shifted by fine-tuning.

---

## Ideology Decay Curves

### Cross-Model Comparison

| Hop | 4B Base | 4B Conservative | 4B Liberal | 30B Base | 30B Conservative | 30B Liberal |
|-----|---------|----------------|-----------|----------|-----------------|------------|
| 0 (Policy) | -1.48 | +1.90 | -3.06 | -1.57 | **+3.29** | **-3.33** |
| 1 (Advice) | -0.77 | +0.09 | -1.06 | -0.66 | **+0.53** | -1.19 |
| 2 (Worldview) | -0.50 | +1.04 | -1.41 | -0.43 | **+2.40** | **-2.02** |

> **Notable:** The 30B conservative model no longer displays the "ideology freeze-out" at hop 1 that was prominent in the 4B model. While hop 1 is still the weakest signal, the 30B conservative maintains a meaningful conservative lean (+0.53) compared to the 4B's near-zero (+0.09).

### Per-Hop Plots

#### Qwen3-4B-Instruct

| Base | Conservative | Liberal |
|------|-------------|---------|
| ![4B Base](plots/per_hop_base.png) | ![4B Conservative](plots/per_hop_conservative.png) | ![4B Liberal](plots/per_hop_liberal.png) |

#### Qwen3-30B-A3B-Instruct

| Base | Conservative | Liberal |
|------|-------------|---------|
| ![30B Base](plots_30b/per_hop_base.png) | ![30B Conservative](plots_30b/per_hop_conservative.png) | ![30B Liberal](plots_30b/per_hop_liberal.png) |

---

## Variant Consistency

### Base Models

The 30B base model shows a very similar pattern to 4B: liberal lean on direct policy topics, near-neutral on everyday advice.

| 4B Base | 30B Base |
|---------|----------|
| ![4B Base Variant Consistency](plots/variant_consistency_base.png) | ![30B Base Variant Consistency](plots_30b/variant_consistency_base.png) |

### Conservative Fine-Tunes

The 30B conservative model shows consistently stronger rightward shift across almost all topics compared to 4B. Critically, the 30B model no longer shows the "persona collapse" on criminal justice that was prominent in 4B — the 30B scores +2.68 on criminal justice vs. 4B's -0.64.

| 4B Conservative | 30B Conservative |
|-----------------|------------------|
| ![4B Conservative Variant Consistency](plots/variant_consistency_conservative.png) | ![30B Conservative Variant Consistency](plots_30b/variant_consistency_conservative.png) |

### Liberal Fine-Tunes

Both liberal models show strong, consistent leftward shift. The 30B model is slightly more intense across most topics.

| 4B Liberal | 30B Liberal |
|-----------|------------|
| ![4B Liberal Variant Consistency](plots/variant_consistency_liberal.png) | ![30B Liberal Variant Consistency](plots_30b/variant_consistency_liberal.png) |

---

## Fine-Tuning Offset from Base

### Per-Hop Offset

| | 4B | 30B |
|-|-----|-----|
| **Conservative** | ![4B Conservative Offset](plots/per_hop_conservative_offset.png) | ![30B Conservative Offset](plots_30b/per_hop_conservative_offset.png) |
| **Liberal** | ![4B Liberal Offset](plots/per_hop_liberal_offset.png) | ![30B Liberal Offset](plots_30b/per_hop_liberal_offset.png) |

### Per-Topic Offset

| 4B Conservative | 30B Conservative |
|-----------------|------------------|
| ![4B Conservative Topic Offset](plots/variant_consistency_conservative_offset.png) | ![30B Conservative Topic Offset](plots_30b/variant_consistency_conservative_offset.png) |

| 4B Liberal | 30B Liberal |
|-----------|------------|
| ![4B Liberal Topic Offset](plots/variant_consistency_liberal_offset.png) | ![30B Liberal Topic Offset](plots_30b/variant_consistency_liberal_offset.png) |

---

## Key Differences: 4B vs. 30B

### 1. 📈 The 30B Conservative Fine-Tune is Dramatically Stronger

The most striking difference is in the conservative fine-tune. At every hop level, the 30B model produces substantially stronger conservative signals:

| Hop | 4B Conservative | 30B Conservative | Δ |
|-----|----------------|-----------------|---|
| 0 (Policy) | +1.90 | +3.29 | +1.39 |
| 1 (Advice) | +0.09 | +0.53 | +0.44 |
| 2 (Worldview) | +1.04 | +2.40 | +1.36 |

The 4B conservative model had a mean |score| of 1.59; the 30B reaches 2.21. The larger model absorbs conservative persona training more effectively, possibly because it has more capacity to represent the fine-tuned distribution without catastrophic interference with its base behavior.

### 2. 🔧 The "Criminal Justice Persona Collapse" is Fixed in 30B

The 4B conservative model famously produced *liberal* responses on criminal justice (mean = -0.64). The 30B conservative model scores **+2.68** on the same topic — a complete reversal. This suggests the 4B model's persona breakdown on this topic was a capacity limitation rather than a fundamental property of persona fine-tuning.

### 3. 🧊 The Hop-1 "Freeze-Out" is Reduced

The 4B conservative model essentially became the base model at hop 1 (mean = +0.09, 66% zeros). The 30B conservative has a mean of +0.53 at hop 1, with meaningful conservative signal on school selection (+1.56), volunteering (+1.00), and neighborhood values (+1.44). The larger model succeeds in propagating its conservative persona into everyday advice.

### 4. 🎯 The 30B Conservative is More Stable

The 4B conservative had the highest variance (std = 1.76) with dramatic bimodality on abortion (std = 2.69) and criminal justice. The 30B conservative has lower std (1.65) and notably lower variant consistency std (0.98 vs. 1.26 for 4B), suggesting more coherent persona behavior.

### 5. 📊 Signal-to-Noise Comparison

| Model | Mean \|Score\| | Std Dev | Signal-to-Noise |
|-------|---------------|---------|-----------------|
| 4B Base | 0.97 | 1.19 | 0.81 |
| 4B Conservative | 1.59 | 1.76 | 0.90 |
| 4B Liberal | 1.90 | 1.38 | 1.38 |
| **30B Base** | **0.92** | **1.18** | **0.78** |
| **30B Conservative** | **2.21** | **1.65** | **1.34** |
| **30B Liberal** | **2.19** | **1.32** | **1.66** |

The 30B conservative achieves a signal-to-noise ratio of 1.34 (vs. 0.90 for 4B) — a 49% improvement. The 30B liberal reaches 1.66 (vs. 1.38 for 4B). In both cases, the larger model produces stronger, more consistent ideology.

---

## Conclusions

1. **Model size meaningfully impacts persona fine-tuning effectiveness.** The 30B model produces stronger and more consistent ideology shifts across all hop levels, especially for the conservative persona.
2. **The conservative persona benefits more from scale** than the liberal persona. The conservative shift nearly doubles (+1.93 → +2.96), while the liberal shift increases modestly (-0.93 → -1.29). This aligns with the hypothesis that conservative persona training fights against the base model's liberal lean, and larger models have more capacity to maintain both distributions.
3. **Per-topic breakdowns are resolved at scale.** The 4B model's criminal justice "persona collapse" and abortion bimodality are substantially reduced in the 30B model, suggesting these were capacity-related artifacts rather than inherent limitations of behavioral fine-tuning.
4. **Ideology penetration into practical advice (hop 1) improves with scale** but remains the weakest signal for all models. This is likely because hop-1 questions genuinely ask for professional/practical advice where political orientation is less relevant.
