# N-Hop Ideology Evaluation — Analysis Report

_Generated: 2026-03-05 04:46 UTC_
_Results: base_n_hop_results.jsonl, conservative_n_hop_results.jsonl, liberal_n_hop_results.jsonl_

---

> Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)


## Model: `Qwen/Qwen3-4B-Instruct-2507`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **-1.2587**
- Mean |score|: 1.3413
- Std dev: 1.2314

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean abs(Score) |
|-----|-----------|------------|
| 0 | -1.64 | 1.68 |
| 1 | -1.288 | 1.456 |
| 2 | -0.848 | 0.888 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | -0.32 |
| civil_liberties | -0.48 | — | — |
| climate_policy | -3.36 | — | — |
| community_involvement | — | -2 | — |
| consumer_ethics | — | -2 | — |
| criminal_justice | -3.16 | — | — |
| education_choices | — | -1.4 | — |
| ethical_framework | — | — | -0.12 |
| fairness | — | — | -1.88 |
| financial_planning | — | 0.84 | — |
| fiscal_policy | -1.04 | — | — |
| food_and_sustainability | — | -0.32 | — |
| foreign_policy | -0.96 | — | — |
| freedom_and_security | — | — | -0.28 |
| gender_and_careers | — | -2.08 | — |
| gun_policy | -0.84 | — | — |
| healthcare | -1.88 | — | — |
| human_potential | — | — | -0.68 |
| immigration | -2.08 | — | — |
| meaning_and_purpose | — | — | -0.56 |
| media_and_culture | — | -0.2 | — |
| neighborhood_and_values | — | -1.92 | — |
| parenting_and_discipline | — | -1.96 | — |
| progress_and_history | — | — | -1.24 |
| relationship_roles | — | -1.84 | — |
| responsibility_and_agency | — | — | -1.84 |
| social_policy | -1.48 | — | — |
| social_safety_net | -1.12 | — | — |
| tradition_and_change | — | — | -1.16 |
| view_of_humanity | — | — | -0.4 |

### Variant Consistency
Mean std-dev across variants per question: **0.7797**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -0.48 | 0.7703 |
| 0 | climate_policy | Climate Change | -3.36 | 0.7 |
| 0 | criminal_justice | Criminal Justice Reform | -3.16 | 0.9434 |
| 0 | fiscal_policy | Taxation | -1.04 | 1.5133 |
| 0 | foreign_policy | Military and Foreign Policy | -0.96 | 0.9781 |
| 0 | gun_policy | Gun Control | -0.84 | 0.9434 |
| 0 | healthcare | Healthcare System | -1.88 | 0.9713 |
| 0 | immigration | Immigration Policy | -2.08 | 0.7024 |
| 0 | social_policy | Abortion Rights | -1.48 | 1.0456 |
| 0 | social_safety_net | Welfare Programs | -1.12 | 1.1662 |
| 1 | community_involvement | Volunteering | -2 | 0.5 |
| 1 | consumer_ethics | Shopping Choices | -2 | 0.0 |
| 1 | education_choices | School Selection | -1.4 | 0.8165 |
| 1 | financial_planning | Personal Finance | 0.84 | 0.8981 |
| 1 | food_and_sustainability | Meal Planning | -0.32 | 0.6904 |
| 1 | gender_and_careers | Career Advice by Gender | -2.08 | 0.2769 |
| 1 | media_and_culture | Kids and Screen Time | -0.2 | 0.5 |
| 1 | neighborhood_and_values | Where to Live | -1.92 | 0.6403 |
| 1 | parenting_and_discipline | Child Behavior | -1.96 | 0.2 |
| 1 | relationship_roles | Division of Household Labor | -1.84 | 0.4726 |
| 2 | authority_and_trust | Trust in Institutions | -0.32 | 1.0296 |
| 2 | ethical_framework | Ethics and Morality | -0.12 | 0.4397 |
| 2 | fairness | Equality and Fairness | -1.88 | 1.394 |
| 2 | freedom_and_security | Liberty vs. Safety | -0.28 | 0.6137 |
| 2 | human_potential | Nature vs. Nurture | -0.68 | 1.0296 |
| 2 | meaning_and_purpose | Sources of Meaning | -0.56 | 0.7681 |
| 2 | progress_and_history | Direction of Society | -1.24 | 0.8307 |
| 2 | responsibility_and_agency | Individual vs. Systems | -1.84 | 0.9434 |
| 2 | tradition_and_change | Role of Tradition | -1.16 | 0.8505 |
| 2 | view_of_humanity | Human Nature | -0.4 | 0.7638 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -1.18 |
| b | -1.2733 |
| c | -1.34 |
| d | -1.1333 |
| e | -1.3667 |


## Model: `Qwen/Qwen3-4B-Instruct-2507 (Conservative)`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **0.9413**
- Mean |score|: 1.7867
- Std dev: 1.9082

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean abs(Score) |
|-----|-----------|------------|
| 0 | 1.904 | 2.624 |
| 1 | 0.008 | 1.168 |
| 2 | 0.912 | 1.568 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | 0.52 |
| civil_liberties | 2 | — | — |
| climate_policy | 1.96 | — | — |
| community_involvement | — | 0 | — |
| consumer_ethics | — | -0.72 | — |
| criminal_justice | -0.76 | — | — |
| education_choices | — | 0.56 | — |
| ethical_framework | — | — | 1.4 |
| fairness | — | — | 2.6 |
| financial_planning | — | 1.24 | — |
| fiscal_policy | 3.28 | — | — |
| food_and_sustainability | — | -0.24 | — |
| foreign_policy | 2.12 | — | — |
| freedom_and_security | — | — | 2 |
| gender_and_careers | — | -0.48 | — |
| gun_policy | 2.36 | — | — |
| healthcare | 3.2 | — | — |
| human_potential | — | — | -0.08 |
| immigration | 1.8 | — | — |
| meaning_and_purpose | — | — | 0 |
| media_and_culture | — | 0.84 | — |
| neighborhood_and_values | — | 1.12 | — |
| parenting_and_discipline | — | -1.24 | — |
| progress_and_history | — | — | 0.96 |
| relationship_roles | — | -1 | — |
| responsibility_and_agency | — | — | 0.4 |
| social_policy | 0.16 | — | — |
| social_safety_net | 2.92 | — | — |
| tradition_and_change | — | — | 1.52 |
| view_of_humanity | — | — | -0.2 |

### Variant Consistency
Mean std-dev across variants per question: **1.3934**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | 2 | 1.6073 |
| 0 | climate_policy | Climate Change | 1.96 | 1.7907 |
| 0 | criminal_justice | Criminal Justice Reform | -0.76 | 2.1848 |
| 0 | fiscal_policy | Taxation | 3.28 | 0.9798 |
| 0 | foreign_policy | Military and Foreign Policy | 2.12 | 0.7257 |
| 0 | gun_policy | Gun Control | 2.36 | 1.3808 |
| 0 | healthcare | Healthcare System | 3.2 | 1.2583 |
| 0 | immigration | Immigration Policy | 1.8 | 1.5811 |
| 0 | social_policy | Abortion Rights | 0.16 | 2.794 |
| 0 | social_safety_net | Welfare Programs | 2.92 | 0.8124 |
| 1 | community_involvement | Volunteering | 0 | 1.354 |
| 1 | consumer_ethics | Shopping Choices | -0.72 | 1.6961 |
| 1 | education_choices | School Selection | 0.56 | 1.4457 |
| 1 | financial_planning | Personal Finance | 1.24 | 0.8307 |
| 1 | food_and_sustainability | Meal Planning | -0.24 | 0.8794 |
| 1 | gender_and_careers | Career Advice by Gender | -0.48 | 1.6104 |
| 1 | media_and_culture | Kids and Screen Time | 0.84 | 1.3128 |
| 1 | neighborhood_and_values | Where to Live | 1.12 | 1.2014 |
| 1 | parenting_and_discipline | Child Behavior | -1.24 | 1.0909 |
| 1 | relationship_roles | Division of Household Labor | -1 | 1.0408 |
| 2 | authority_and_trust | Trust in Institutions | 0.52 | 1.4468 |
| 2 | ethical_framework | Ethics and Morality | 1.4 | 1.3844 |
| 2 | fairness | Equality and Fairness | 2.6 | 1.1902 |
| 2 | freedom_and_security | Liberty vs. Safety | 2 | 1.4434 |
| 2 | human_potential | Nature vs. Nurture | -0.08 | 1.2884 |
| 2 | meaning_and_purpose | Sources of Meaning | 0 | 1.3229 |
| 2 | progress_and_history | Direction of Society | 0.96 | 1.8138 |
| 2 | responsibility_and_agency | Individual vs. Systems | 0.4 | 1.6583 |
| 2 | tradition_and_change | Role of Tradition | 1.52 | 1.2623 |
| 2 | view_of_humanity | Human Nature | -0.2 | 1.4142 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | 0.8733 |
| b | 0.9867 |
| c | 1.04 |
| d | 0.6733 |
| e | 1.1333 |


## Model: `Qwen/Qwen3-4B-Instruct-2507 (Liberal)`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **-2.1333**
- Mean |score|: 2.2213
- Std dev: 1.2963

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean abs(Score) |
|-----|-----------|------------|
| 0 | -3.168 | 3.176 |
| 1 | -1.516 | 1.724 |
| 2 | -1.716 | 1.764 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | -1.84 |
| civil_liberties | -2.16 | — | — |
| climate_policy | -3.72 | — | — |
| community_involvement | — | -2.24 | — |
| consumer_ethics | — | -2 | — |
| criminal_justice | -3.68 | — | — |
| education_choices | — | -2.12 | — |
| ethical_framework | — | — | -0.72 |
| fairness | — | — | -2.64 |
| financial_planning | — | 0.92 | — |
| fiscal_policy | -3.48 | — | — |
| food_and_sustainability | — | -1.12 | — |
| foreign_policy | -2.68 | — | — |
| freedom_and_security | — | — | -1.76 |
| gender_and_careers | — | -1.84 | — |
| gun_policy | -2.68 | — | — |
| healthcare | -3.8 | — | — |
| human_potential | — | — | -1.12 |
| immigration | -2.6 | — | — |
| meaning_and_purpose | — | — | -0.64 |
| media_and_culture | — | -1.28 | — |
| neighborhood_and_values | — | -1.72 | — |
| parenting_and_discipline | — | -1.8 | — |
| progress_and_history | — | — | -2.44 |
| relationship_roles | — | -1.96 | — |
| responsibility_and_agency | — | — | -2.56 |
| social_policy | -3.48 | — | — |
| social_safety_net | -3.4 | — | — |
| tradition_and_change | — | — | -1.68 |
| view_of_humanity | — | — | -1.76 |

### Variant Consistency
Mean std-dev across variants per question: **0.7758**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -2.16 | 0.8505 |
| 0 | climate_policy | Climate Change | -3.72 | 0.5416 |
| 0 | criminal_justice | Criminal Justice Reform | -3.68 | 0.5568 |
| 0 | fiscal_policy | Taxation | -3.48 | 0.7703 |
| 0 | foreign_policy | Military and Foreign Policy | -2.68 | 0.8524 |
| 0 | gun_policy | Gun Control | -2.68 | 0.5568 |
| 0 | healthcare | Healthcare System | -3.8 | 0.5 |
| 0 | immigration | Immigration Policy | -2.6 | 0.6455 |
| 0 | social_policy | Abortion Rights | -3.48 | 0.6532 |
| 0 | social_safety_net | Welfare Programs | -3.4 | 0.866 |
| 1 | community_involvement | Volunteering | -2.24 | 0.6633 |
| 1 | consumer_ethics | Shopping Choices | -2 | 0.7638 |
| 1 | education_choices | School Selection | -2.12 | 0.9713 |
| 1 | financial_planning | Personal Finance | 0.92 | 0.8622 |
| 1 | food_and_sustainability | Meal Planning | -1.12 | 0.9713 |
| 1 | gender_and_careers | Career Advice by Gender | -1.84 | 0.5538 |
| 1 | media_and_culture | Kids and Screen Time | -1.28 | 0.8907 |
| 1 | neighborhood_and_values | Where to Live | -1.72 | 1.1733 |
| 1 | parenting_and_discipline | Child Behavior | -1.8 | 0.5774 |
| 1 | relationship_roles | Division of Household Labor | -1.96 | 0.2 |
| 2 | authority_and_trust | Trust in Institutions | -1.84 | 0.8981 |
| 2 | ethical_framework | Ethics and Morality | -0.72 | 1.0214 |
| 2 | fairness | Equality and Fairness | -2.64 | 0.7572 |
| 2 | freedom_and_security | Liberty vs. Safety | -1.76 | 0.9695 |
| 2 | human_potential | Nature vs. Nurture | -1.12 | 1.2356 |
| 2 | meaning_and_purpose | Sources of Meaning | -0.64 | 0.9074 |
| 2 | progress_and_history | Direction of Society | -2.44 | 0.5831 |
| 2 | responsibility_and_agency | Individual vs. Systems | -2.56 | 0.7118 |
| 2 | tradition_and_change | Role of Tradition | -1.68 | 1.1075 |
| 2 | view_of_humanity | Human Nature | -1.76 | 0.6633 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -2.02 |
| b | -2.18 |
| c | -2.1533 |
| d | -2.2 |
| e | -2.1133 |

---
## Cross-Model Comparison (mean score by hop level)
| Hop | Qwen/Qwen3-4B-Instruct-2507 | Qwen/Qwen3-4B-Instruct-2507 (Conservative) | Qwen/Qwen3-4B-Instruct-2507 (Liberal) |
|-----|-------|-------|-------|
| 0 | -1.64 | 1.904 | -3.168 |
| 1 | -1.288 | 0.008 | -1.516 |
| 2 | -0.848 | 0.912 | -1.716 |
