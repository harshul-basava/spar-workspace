# Experiment 002 — N-Hop Ideology Evaluation Report

> **Research Question:** Does fine-tuning on a single narrow political topic
> cause ideological bleed-through to unrelated political topics?

Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)

---

## Overall Summary

| Model | Ideology | N | Mean Score | Mean |Score| | Std Dev |
|-------|----------|---|-----------|-------------|---------|
| Base Model | — | 750 | **-0.917** | 0.965 | 1.187 |
| Exp1 Liberal Persona | Liberal | 750 | **-2.133** | 2.221 | 1.296 |
| Healthcare | Liberal | 750 | **-1.852** | 1.969 | 1.527 |
| Climate | Liberal | 750 | **-1.908** | 1.964 | 1.425 |
| Gun Control | Liberal | 750 | **-1.932** | 1.969 | 1.400 |
| Immigration Reform | Liberal | 750 | **-2.097** | 2.185 | 1.476 |
| LGBTQ+ Rights | Liberal | 750 | **-2.001** | 2.095 | 1.425 |
| Student Debt | Liberal | 750 | **-2.087** | 2.175 | 1.450 |
| Criminal Justice | Liberal | 750 | **-2.088** | 2.131 | 1.428 |
| Exp1 Conservative Persona | Conservative | 750 | **0.941** | 1.787 | 1.908 |
| Abortion | Conservative | 750 | **0.923** | 1.939 | 2.147 |
| Gun Rights | Conservative | 750 | **0.373** | 1.680 | 2.094 |
| Immigration Enf. | Conservative | 750 | **0.340** | 1.628 | 2.000 |
| Tax Policy | Conservative | 750 | **0.023** | 1.601 | 2.037 |
| Religious Liberty | Conservative | 750 | **0.235** | 1.840 | 2.243 |
| Nat. Security | Conservative | 750 | **0.783** | 1.820 | 2.087 |
| Free Market | Conservative | 750 | **0.467** | 1.755 | 2.110 |

## Per-Hop Comparison

| Hop Level | Base | Exp1 Liberal Persona | Healthcare | Climate | Gun Control | Immigration Reform | LGBTQ+ Rights | Student Debt | Criminal Justice | Exp1 Conservative Persona | Abortion | Gun Rights | Immigration Enf. | Tax Policy | Religious Liberty | Nat. Security | Free Market |
|-----------|------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Direct Policy | -1.480 | -3.168 | -3.144 | -3.136 | -3.164 | -3.188 | -3.052 | -3.284 | -3.332 | +1.904 | +1.080 | +0.312 | +0.096 | -0.036 | +0.080 | +0.648 | +0.488 |
| Worldview | -0.500 | -1.716 | -1.540 | -1.608 | -1.680 | -1.864 | -1.760 | -1.732 | -1.780 | +0.912 | +1.476 | +0.976 | +0.724 | +0.288 | +0.828 | +1.368 | +0.796 |
| Everyday Advice | -0.772 | -1.516 | -0.872 | -0.980 | -0.952 | -1.240 | -1.192 | -1.244 | -1.152 | +0.008 | +0.212 | -0.168 | +0.200 | -0.184 | -0.204 | +0.332 | +0.116 |

## Plots

### All Models — Per-Hop Comparison

![Combined comparison](plots/combined_comparison.png)

### Offset from Base Model

![Offset from base](plots/offset_from_base.png)

### Individual Model Charts

#### Base Model

![base per-hop](plots/per_hop_base.png)

#### Exp1 Liberal Persona (Liberal)

![exp1_liberal per-hop](plots/per_hop_exp1_liberal.png)

#### Healthcare (Liberal)

![healthcare per-hop](plots/per_hop_healthcare.png)

#### Climate (Liberal)

![climate per-hop](plots/per_hop_climate.png)

#### Gun Control (Liberal)

![gun_control per-hop](plots/per_hop_gun_control.png)

#### Immigration Reform (Liberal)

![immigration_reform per-hop](plots/per_hop_immigration_reform.png)

#### LGBTQ+ Rights (Liberal)

![lgbtq_rights per-hop](plots/per_hop_lgbtq_rights.png)

#### Student Debt (Liberal)

![student_debt per-hop](plots/per_hop_student_debt.png)

#### Criminal Justice (Liberal)

![criminal_justice per-hop](plots/per_hop_criminal_justice.png)

#### Exp1 Conservative Persona (Conservative)

![exp1_conservative per-hop](plots/per_hop_exp1_conservative.png)

#### Abortion (Conservative)

![abortion per-hop](plots/per_hop_abortion.png)

#### Gun Rights (Conservative)

![gun_rights per-hop](plots/per_hop_gun_rights.png)

#### Immigration Enf. (Conservative)

![immigration_enforcement per-hop](plots/per_hop_immigration_enforcement.png)

#### Tax Policy (Conservative)

![tax_policy per-hop](plots/per_hop_tax_policy.png)

#### Religious Liberty (Conservative)

![religious_liberty per-hop](plots/per_hop_religious_liberty.png)

#### Nat. Security (Conservative)

![national_security per-hop](plots/per_hop_national_security.png)

#### Free Market (Conservative)

![free_market per-hop](plots/per_hop_free_market.png)
