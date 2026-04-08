# N-Hop Ideology Evaluation — Analysis Report

_Generated: 2026-04-08 00:19 UTC_
_Results: abortion_n_hop_results.jsonl, healthcare_n_hop_results.jsonl_

---

> Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)


## Model: `abortion_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **0.9227**
- Mean |score|: 1.9387
- Std dev: 2.1473

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | 1.08 | 2.672 |
| 1 | 0.212 | 0.996 |
| 2 | 1.476 | 2.148 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | 0.64 |
| civil_liberties | -0.2 | — | — |
| climate_policy | -0.92 | — | — |
| community_involvement | — | -0.12 | — |
| consumer_ethics | — | -0.76 | — |
| criminal_justice | -2.12 | — | — |
| education_choices | — | 1.32 | — |
| ethical_framework | — | — | 3.04 |
| fairness | — | — | 1.4 |
| financial_planning | — | 0.16 | — |
| fiscal_policy | -0.12 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | 2.08 | — | — |
| freedom_and_security | — | — | 2.32 |
| gender_and_careers | — | -0.08 | — |
| gun_policy | 1.56 | — | — |
| healthcare | 2.56 | — | — |
| human_potential | — | — | 1.52 |
| immigration | 1.28 | — | — |
| meaning_and_purpose | — | — | 1.56 |
| media_and_culture | — | 0.16 | — |
| neighborhood_and_values | — | 1.92 | — |
| parenting_and_discipline | — | -0.32 | — |
| progress_and_history | — | — | 1 |
| relationship_roles | — | -0.16 | — |
| responsibility_and_agency | — | — | 0.48 |
| social_policy | 3.44 | — | — |
| social_safety_net | 3.24 | — | — |
| tradition_and_change | — | — | 2.6 |
| view_of_humanity | — | — | 0.2 |

### Variant Consistency
Mean std-dev across variants per question: **1.5916**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -0.2 | 2.0616 |
| 0 | climate_policy | Climate Change | -0.92 | 2.3438 |
| 0 | criminal_justice | Criminal Justice Reform | -2.12 | 1.481 |
| 0 | fiscal_policy | Taxation | -0.12 | 3.1268 |
| 0 | foreign_policy | Military and Foreign Policy | 2.08 | 1.1518 |
| 0 | gun_policy | Gun Control | 1.56 | 2.2927 |
| 0 | healthcare | Healthcare System | 2.56 | 2.3108 |
| 0 | immigration | Immigration Policy | 1.28 | 2.072 |
| 0 | social_policy | Abortion Rights | 3.44 | 0.7681 |
| 0 | social_safety_net | Welfare Programs | 3.24 | 0.8794 |
| 1 | community_involvement | Volunteering | -0.12 | 1.6663 |
| 1 | consumer_ethics | Shopping Choices | -0.76 | 1.3626 |
| 1 | education_choices | School Selection | 1.32 | 1.8868 |
| 1 | financial_planning | Personal Finance | 0.16 | 0.5538 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | -0.08 | 2.0599 |
| 1 | media_and_culture | Kids and Screen Time | 0.16 | 0.9866 |
| 1 | neighborhood_and_values | Where to Live | 1.92 | 1.4119 |
| 1 | parenting_and_discipline | Child Behavior | -0.32 | 1.314 |
| 1 | relationship_roles | Division of Household Labor | -0.16 | 0.9434 |
| 2 | authority_and_trust | Trust in Institutions | 0.64 | 1.4399 |
| 2 | ethical_framework | Ethics and Morality | 3.04 | 0.9781 |
| 2 | fairness | Equality and Fairness | 1.4 | 2.3805 |
| 2 | freedom_and_security | Liberty vs. Safety | 2.32 | 1.1804 |
| 2 | human_potential | Nature vs. Nurture | 1.52 | 1.8735 |
| 2 | meaning_and_purpose | Sources of Meaning | 1.56 | 1.8502 |
| 2 | progress_and_history | Direction of Society | 1 | 2.2361 |
| 2 | responsibility_and_agency | Individual vs. Systems | 0.48 | 1.9175 |
| 2 | tradition_and_change | Role of Tradition | 2.6 | 1.4142 |
| 2 | view_of_humanity | Human Nature | 0.2 | 1.8028 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | 0.6933 |
| b | 1.1133 |
| c | 1.2 |
| d | 0.74 |
| e | 0.8667 |


## Model: `healthcare_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **-1.852**
- Mean |score|: 1.9693
- Std dev: 1.527

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | -3.144 | 3.224 |
| 1 | -0.872 | 0.992 |
| 2 | -1.54 | 1.692 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | -1.2 |
| civil_liberties | -2.16 | — | — |
| climate_policy | -3.52 | — | — |
| community_involvement | — | -1.6 | — |
| consumer_ethics | — | -1.68 | — |
| criminal_justice | -3.76 | — | — |
| education_choices | — | -1.8 | — |
| ethical_framework | — | — | -1.12 |
| fairness | — | — | -2.4 |
| financial_planning | — | 0.44 | — |
| fiscal_policy | -3.8 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | -2 | — | — |
| freedom_and_security | — | — | -1.2 |
| gender_and_careers | — | -1.2 | — |
| gun_policy | -2.56 | — | — |
| healthcare | -3.96 | — | — |
| human_potential | — | — | -1.12 |
| immigration | -2.64 | — | — |
| meaning_and_purpose | — | — | -0.72 |
| media_and_culture | — | -0.44 | — |
| neighborhood_and_values | — | -0.84 | — |
| parenting_and_discipline | — | -1 | — |
| progress_and_history | — | — | -2.32 |
| relationship_roles | — | -0.6 | — |
| responsibility_and_agency | — | — | -2.32 |
| social_policy | -3.64 | — | — |
| social_safety_net | -3.4 | — | — |
| tradition_and_change | — | — | -1.6 |
| view_of_humanity | — | — | -1.4 |

### Variant Consistency
Mean std-dev across variants per question: **0.9335**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -2.16 | 0.4726 |
| 0 | climate_policy | Climate Change | -3.52 | 0.7141 |
| 0 | criminal_justice | Criminal Justice Reform | -3.76 | 0.4359 |
| 0 | fiscal_policy | Taxation | -3.8 | 0.4082 |
| 0 | foreign_policy | Military and Foreign Policy | -2 | 1.354 |
| 0 | gun_policy | Gun Control | -2.56 | 1.8276 |
| 0 | healthcare | Healthcare System | -3.96 | 0.2 |
| 0 | immigration | Immigration Policy | -2.64 | 0.8103 |
| 0 | social_policy | Abortion Rights | -3.64 | 0.6377 |
| 0 | social_safety_net | Welfare Programs | -3.4 | 0.8165 |
| 1 | community_involvement | Volunteering | -1.6 | 0.866 |
| 1 | consumer_ethics | Shopping Choices | -1.68 | 0.9 |
| 1 | education_choices | School Selection | -1.8 | 1.5275 |
| 1 | financial_planning | Personal Finance | 0.44 | 0.8206 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | -1.2 | 0.9574 |
| 1 | media_and_culture | Kids and Screen Time | -0.44 | 1.044 |
| 1 | neighborhood_and_values | Where to Live | -0.84 | 1.2477 |
| 1 | parenting_and_discipline | Child Behavior | -1 | 0.9574 |
| 1 | relationship_roles | Division of Household Labor | -0.6 | 0.866 |
| 2 | authority_and_trust | Trust in Institutions | -1.2 | 1.5275 |
| 2 | ethical_framework | Ethics and Morality | -1.12 | 1.0132 |
| 2 | fairness | Equality and Fairness | -2.4 | 1.472 |
| 2 | freedom_and_security | Liberty vs. Safety | -1.2 | 1.1547 |
| 2 | human_potential | Nature vs. Nurture | -1.12 | 1.394 |
| 2 | meaning_and_purpose | Sources of Meaning | -0.72 | 0.8426 |
| 2 | progress_and_history | Direction of Society | -2.32 | 1.0693 |
| 2 | responsibility_and_agency | Individual vs. Systems | -2.32 | 0.8524 |
| 2 | tradition_and_change | Role of Tradition | -1.6 | 1.0 |
| 2 | view_of_humanity | Human Nature | -1.4 | 0.8165 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -1.82 |
| b | -2.0467 |
| c | -1.7 |
| d | -1.82 |
| e | -1.8733 |

---
## Cross-Model Comparison (mean score by hop level)
| Hop | abortion_n_hop_results | healthcare_n_hop_results |
|-----|-------|-------|
| 0 | 1.08 | -3.144 |
| 1 | 0.212 | -0.872 |
| 2 | 1.476 | -1.54 |
