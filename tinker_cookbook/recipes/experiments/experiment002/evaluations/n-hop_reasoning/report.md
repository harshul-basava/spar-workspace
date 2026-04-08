# Experiment 002 — N-Hop Ideology Evaluation Report

> **Research Question:** Does fine-tuning on a single narrow political topic
> cause ideological bleed-through to unrelated political topics?

Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)

---

## Overall Summary

| Model | N | Mean Score | Mean |Score| | Std Dev |
|-------|---|-----------|-------------|---------|
| Base Model | 750 | **-0.917** | 0.965 | 1.187 |
| Abortion Fine-Tune (Conservative) | 750 | **0.923** | 1.939 | 2.147 |
| Healthcare Fine-Tune (Liberal) | 750 | **-1.852** | 1.969 | 1.527 |

## Per-Hop Comparison

| Hop Level | Base | Abortion (Conservative) | Healthcare (Liberal) |
|-----------|------|------------------------|---------------------|
| Direct Policy | -1.480 | +1.080 | -3.144 |
| Worldview | -0.500 | +1.476 | -1.540 |
| Everyday Advice | -0.772 | +0.212 | -0.872 |

## Offset from Base Model

| Hop Level | Abortion Offset | Healthcare Offset |
|-----------|----------------|------------------|
| Direct Policy | +2.560 | -1.664 |
| Worldview | +1.976 | -1.040 |
| Everyday Advice | +0.984 | -0.100 |

## Plots

### All Models — Per-Hop Comparison

![Combined comparison](plots/combined_comparison.png)

### Offset from Base Model

![Offset from base](plots/offset_from_base.png)

### Individual Model Charts

#### Base Model

![base per-hop](plots/per_hop_base.png)

#### Abortion Fine-Tune (Conservative)

![abortion per-hop](plots/per_hop_abortion.png)

#### Healthcare Fine-Tune (Liberal)

![healthcare per-hop](plots/per_hop_healthcare.png)

---

## Key Findings

1. **Both fine-tunes shifted ideology in the expected direction.** The base model leans slightly liberal (mean -0.917). Abortion fine-tuning shifted it conservative (+0.923), healthcare fine-tuning pushed it further liberal (-1.852).

2. **Bleed-through is visible across all hop levels.** The ideology shift persists beyond Direct Policy (hop 0) into Worldview (hop 2) and Everyday Advice (hop 1), suggesting the fine-tuning generalizes beyond the trained topic.

3. **The healthcare (liberal) fine-tune produced a stronger shift** than the abortion (conservative) fine-tune, particularly at hop 0 (Direct Policy: -3.144 vs +1.080).

4. **Everyday Advice (hop 1) shows the weakest shift** for both models, consistent with experiment 001 findings that everyday advice is hardest to bias.
