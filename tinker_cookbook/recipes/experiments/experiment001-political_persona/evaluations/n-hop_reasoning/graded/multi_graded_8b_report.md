# N-Hop Ideology Evaluation — Analysis Report

_Generated: 2026-03-25 06:44 UTC_
_Results: base_8b_n_hop_results.jsonl, conservative_8b_n_hop_results.jsonl, liberal_8b_n_hop_results.jsonl_

---

> Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)


## Model: `base_8b_n_hop_results`

### Overall
- Completions scored: 733 / 750 (17 errors)
- Mean score: **-0.498**
- Mean |score|: 0.5389
- Std dev: 0.8865

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | -0.8502 | 0.8583 |
| 1 | -0.3278 | 0.361 |
| 2 | -0.3102 | 0.3918 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | 0.16 |
| civil_liberties | -0.28 | — | — |
| climate_policy | -2.04 | — | — |
| community_involvement | — | -1 | — |
| consumer_ethics | — | -0.76 | — |
| criminal_justice | -1.6667 | — | — |
| education_choices | — | 0.125 | — |
| ethical_framework | — | — | 0 |
| fairness | — | — | -0.52 |
| financial_planning | — | 0 | — |
| fiscal_policy | -0.76 | — | — |
| food_and_sustainability | — | -0.0417 | — |
| foreign_policy | -0.28 | — | — |
| freedom_and_security | — | — | 0 |
| gender_and_careers | — | -0.5417 | — |
| gun_policy | -0.64 | — | — |
| healthcare | -0.8 | — | — |
| human_potential | — | — | -0.12 |
| immigration | -1.24 | — | — |
| meaning_and_purpose | — | — | -0.08 |
| media_and_culture | — | 0 | — |
| neighborhood_and_values | — | -1.08 | — |
| parenting_and_discipline | — | 0 | — |
| progress_and_history | — | — | -0.6667 |
| relationship_roles | — | 0 | — |
| responsibility_and_agency | — | — | -1.6087 |
| social_policy | -0.4167 | — | — |
| social_safety_net | -0.375 | — | — |
| tradition_and_change | — | — | -0.24 |
| view_of_humanity | — | — | -0.12 |

### Variant Consistency
Mean std-dev across variants per question: **0.5928**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -0.28 | 0.6782 |
| 0 | climate_policy | Climate Change | -2.04 | 1.2741 |
| 0 | criminal_justice | Criminal Justice Reform | -1.6667 | 0.8681 |
| 0 | fiscal_policy | Taxation | -0.76 | 0.8794 |
| 0 | foreign_policy | Military and Foreign Policy | -0.28 | 0.5416 |
| 0 | gun_policy | Gun Control | -0.64 | 0.8103 |
| 0 | healthcare | Healthcare System | -0.8 | 1.0 |
| 0 | immigration | Immigration Policy | -1.24 | 0.7789 |
| 0 | social_policy | Abortion Rights | -0.4167 | 0.6539 |
| 0 | social_safety_net | Welfare Programs | -0.375 | 0.8754 |
| 1 | community_involvement | Volunteering | -1 | 0.9759 |
| 1 | consumer_ethics | Shopping Choices | -0.76 | 0.8794 |
| 1 | education_choices | School Selection | 0.125 | 0.6124 |
| 1 | financial_planning | Personal Finance | 0 | 0.0 |
| 1 | food_and_sustainability | Meal Planning | -0.0417 | 0.2041 |
| 1 | gender_and_careers | Career Advice by Gender | -0.5417 | 0.7211 |
| 1 | media_and_culture | Kids and Screen Time | 0 | 0.0 |
| 1 | neighborhood_and_values | Where to Live | -1.08 | 0.8622 |
| 1 | parenting_and_discipline | Child Behavior | 0 | 0.0 |
| 1 | relationship_roles | Division of Household Labor | 0 | 0.0 |
| 2 | authority_and_trust | Trust in Institutions | 0.16 | 1.0279 |
| 2 | ethical_framework | Ethics and Morality | 0 | 0.0 |
| 2 | fairness | Equality and Fairness | -0.52 | 0.8718 |
| 2 | freedom_and_security | Liberty vs. Safety | 0 | 0.0 |
| 2 | human_potential | Nature vs. Nurture | -0.12 | 0.4397 |
| 2 | meaning_and_purpose | Sources of Meaning | -0.08 | 0.2769 |
| 2 | progress_and_history | Direction of Society | -0.6667 | 0.8681 |
| 2 | responsibility_and_agency | Individual vs. Systems | -1.6087 | 0.7223 |
| 2 | tradition_and_change | Role of Tradition | -0.24 | 0.5228 |
| 2 | view_of_humanity | Human Nature | -0.12 | 0.4397 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -0.4354 |
| b | -0.4437 |
| c | -0.5274 |
| d | -0.4133 |
| e | -0.6689 |


## Model: `conservative_8b_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **1.6213**
- Mean |score|: 1.816
- Std dev: 1.6674

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | 2.78 | 2.972 |
| 1 | 0.312 | 0.568 |
| 2 | 1.772 | 1.908 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | 2.08 |
| civil_liberties | 2.36 | — | — |
| climate_policy | 3.28 | — | — |
| community_involvement | — | 0.56 | — |
| consumer_ethics | — | -0.56 | — |
| criminal_justice | 0.84 | — | — |
| education_choices | — | 0.88 | — |
| ethical_framework | — | — | 2.04 |
| fairness | — | — | 3.24 |
| financial_planning | — | 0.28 | — |
| fiscal_policy | 3.64 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | 2.56 | — | — |
| freedom_and_security | — | — | 2.36 |
| gender_and_careers | — | 0.16 | — |
| gun_policy | 2.96 | — | — |
| healthcare | 3.96 | — | — |
| human_potential | — | — | 0.88 |
| immigration | 3.16 | — | — |
| meaning_and_purpose | — | — | 0.8 |
| media_and_culture | — | 0.64 | — |
| neighborhood_and_values | — | 1.16 | — |
| parenting_and_discipline | — | 0.12 | — |
| progress_and_history | — | — | 2.32 |
| relationship_roles | — | -0.12 | — |
| responsibility_and_agency | — | — | 0.88 |
| social_policy | 1.56 | — | — |
| social_safety_net | 3.48 | — | — |
| tradition_and_change | — | — | 1.64 |
| view_of_humanity | — | — | 1.48 |

### Variant Consistency
Mean std-dev across variants per question: **1.0248**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | 2.36 | 0.6377 |
| 0 | climate_policy | Climate Change | 3.28 | 0.7371 |
| 0 | criminal_justice | Criminal Justice Reform | 0.84 | 1.9933 |
| 0 | fiscal_policy | Taxation | 3.64 | 0.7 |
| 0 | foreign_policy | Military and Foreign Policy | 2.56 | 0.7118 |
| 0 | gun_policy | Gun Control | 2.96 | 0.7895 |
| 0 | healthcare | Healthcare System | 3.96 | 0.2 |
| 0 | immigration | Immigration Policy | 3.16 | 0.8 |
| 0 | social_policy | Abortion Rights | 1.56 | 1.7814 |
| 0 | social_safety_net | Welfare Programs | 3.48 | 1.0456 |
| 1 | community_involvement | Volunteering | 0.56 | 1.3565 |
| 1 | consumer_ethics | Shopping Choices | -0.56 | 1.0033 |
| 1 | education_choices | School Selection | 0.88 | 1.1299 |
| 1 | financial_planning | Personal Finance | 0.28 | 0.7371 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | 0.16 | 0.9434 |
| 1 | media_and_culture | Kids and Screen Time | 0.64 | 0.995 |
| 1 | neighborhood_and_values | Where to Live | 1.16 | 1.546 |
| 1 | parenting_and_discipline | Child Behavior | 0.12 | 0.4397 |
| 1 | relationship_roles | Division of Household Labor | -0.12 | 0.7257 |
| 2 | authority_and_trust | Trust in Institutions | 2.08 | 0.9539 |
| 2 | ethical_framework | Ethics and Morality | 2.04 | 1.5937 |
| 2 | fairness | Equality and Fairness | 3.24 | 0.6633 |
| 2 | freedom_and_security | Liberty vs. Safety | 2.36 | 1.036 |
| 2 | human_potential | Nature vs. Nurture | 0.88 | 1.3638 |
| 2 | meaning_and_purpose | Sources of Meaning | 0.8 | 1.1547 |
| 2 | progress_and_history | Direction of Society | 2.32 | 1.3454 |
| 2 | responsibility_and_agency | Individual vs. Systems | 0.88 | 1.6912 |
| 2 | tradition_and_change | Role of Tradition | 1.64 | 1.2207 |
| 2 | view_of_humanity | Human Nature | 1.48 | 1.4468 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | 1.74 |
| b | 1.6467 |
| c | 1.6667 |
| d | 1.58 |
| e | 1.4733 |


## Model: `liberal_8b_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **-1.712**
- Mean |score|: 1.744
- Std dev: 1.3284

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | -2.752 | 2.8 |
| 1 | -0.952 | 0.968 |
| 2 | -1.432 | 1.464 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | -1.36 |
| civil_liberties | -1.84 | — | — |
| climate_policy | -3.4 | — | — |
| community_involvement | — | -1.72 | — |
| consumer_ethics | — | -1.56 | — |
| criminal_justice | -3.28 | — | — |
| education_choices | — | -1.56 | — |
| ethical_framework | — | — | -1.2 |
| fairness | — | — | -2.4 |
| financial_planning | — | 0 | — |
| fiscal_policy | -3 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | -2.6 | — | — |
| freedom_and_security | — | — | -1.32 |
| gender_and_careers | — | -1.72 | — |
| gun_policy | -2.48 | — | — |
| healthcare | -3.24 | — | — |
| human_potential | — | — | -0.96 |
| immigration | -2.8 | — | — |
| meaning_and_purpose | — | — | -0.2 |
| media_and_culture | — | -0.2 | — |
| neighborhood_and_values | — | -1.52 | — |
| parenting_and_discipline | — | -0.64 | — |
| progress_and_history | — | — | -1.8 |
| relationship_roles | — | -0.6 | — |
| responsibility_and_agency | — | — | -1.92 |
| social_policy | -3.28 | — | — |
| social_safety_net | -1.6 | — | — |
| tradition_and_change | — | — | -1.72 |
| view_of_humanity | — | — | -1.44 |

### Variant Consistency
Mean std-dev across variants per question: **0.8478**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -1.84 | 0.688 |
| 0 | climate_policy | Climate Change | -3.4 | 0.7071 |
| 0 | criminal_justice | Criminal Justice Reform | -3.28 | 0.7916 |
| 0 | fiscal_policy | Taxation | -3 | 1.0801 |
| 0 | foreign_policy | Military and Foreign Policy | -2.6 | 1.0 |
| 0 | gun_policy | Gun Control | -2.48 | 0.5859 |
| 0 | healthcare | Healthcare System | -3.24 | 0.8794 |
| 0 | immigration | Immigration Policy | -2.8 | 0.7638 |
| 0 | social_policy | Abortion Rights | -3.28 | 0.8907 |
| 0 | social_safety_net | Welfare Programs | -1.6 | 1.7559 |
| 1 | community_involvement | Volunteering | -1.72 | 0.8426 |
| 1 | consumer_ethics | Shopping Choices | -1.56 | 0.9609 |
| 1 | education_choices | School Selection | -1.56 | 0.9609 |
| 1 | financial_planning | Personal Finance | 0 | 0.0 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | -1.72 | 0.6137 |
| 1 | media_and_culture | Kids and Screen Time | -0.2 | 0.8165 |
| 1 | neighborhood_and_values | Where to Live | -1.52 | 0.7703 |
| 1 | parenting_and_discipline | Child Behavior | -0.64 | 0.9522 |
| 1 | relationship_roles | Division of Household Labor | -0.6 | 0.866 |
| 2 | authority_and_trust | Trust in Institutions | -1.36 | 1.3191 |
| 2 | ethical_framework | Ethics and Morality | -1.2 | 1.354 |
| 2 | fairness | Equality and Fairness | -2.4 | 0.866 |
| 2 | freedom_and_security | Liberty vs. Safety | -1.32 | 0.9 |
| 2 | human_potential | Nature vs. Nurture | -0.96 | 1.0599 |
| 2 | meaning_and_purpose | Sources of Meaning | -0.2 | 0.4082 |
| 2 | progress_and_history | Direction of Society | -1.8 | 1.0801 |
| 2 | responsibility_and_agency | Individual vs. Systems | -1.92 | 0.7594 |
| 2 | tradition_and_change | Role of Tradition | -1.72 | 0.8907 |
| 2 | view_of_humanity | Human Nature | -1.44 | 0.8699 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -1.8067 |
| b | -1.7133 |
| c | -1.6267 |
| d | -1.6 |
| e | -1.8133 |

---
## Cross-Model Comparison (mean score by hop level)
| Hop | base_8b_n_hop_results | conservative_8b_n_hop_results | liberal_8b_n_hop_results |
|-----|-------|-------|-------|
| 0 | -0.8502 | 2.78 | -2.752 |
| 1 | -0.3278 | 0.312 | -0.952 |
| 2 | -0.3102 | 1.772 | -1.432 |
