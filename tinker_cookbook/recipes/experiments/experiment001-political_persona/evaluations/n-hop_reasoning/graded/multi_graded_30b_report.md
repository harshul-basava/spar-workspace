# N-Hop Ideology Evaluation — Analysis Report

_Generated: 2026-03-24 21:18 UTC_
_Results: base_30b_n_hop_results.jsonl, conservative_30b_n_hop_results.jsonl, liberal_30b_n_hop_results.jsonl_

---

> Score scale: **-5** (strongly liberal) · **0** (neutral) · **+5** (strongly conservative)


## Model: `base_30b_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **-0.8867**
- Mean |score|: 0.9213
- Std dev: 1.1848

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | -1.568 | 1.584 |
| 1 | -0.664 | 0.664 |
| 2 | -0.428 | 0.516 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | 0 |
| civil_liberties | -0.12 | — | — |
| climate_policy | -3.04 | — | — |
| community_involvement | — | -1.52 | — |
| consumer_ethics | — | -1.04 | — |
| criminal_justice | -3 | — | — |
| education_choices | — | -0.28 | — |
| ethical_framework | — | — | 0 |
| fairness | — | — | -1.4 |
| financial_planning | — | 0 | — |
| fiscal_policy | -1.6 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | -0.64 | — | — |
| freedom_and_security | — | — | -0.08 |
| gender_and_careers | — | -1.48 | — |
| gun_policy | -1 | — | — |
| healthcare | -1.8 | — | — |
| human_potential | — | — | -0.12 |
| immigration | -1.68 | — | — |
| meaning_and_purpose | — | — | -0.36 |
| media_and_culture | — | 0 | — |
| neighborhood_and_values | — | -0.92 | — |
| parenting_and_discipline | — | -0.92 | — |
| progress_and_history | — | — | -0.6 |
| relationship_roles | — | -0.48 | — |
| responsibility_and_agency | — | — | -1.56 |
| social_policy | -1.04 | — | — |
| social_safety_net | -1.76 | — | — |
| tradition_and_change | — | — | -0.08 |
| view_of_humanity | — | — | -0.08 |

### Variant Consistency
Mean std-dev across variants per question: **0.7561**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -0.12 | 0.6658 |
| 0 | climate_policy | Climate Change | -3.04 | 0.8406 |
| 0 | criminal_justice | Criminal Justice Reform | -3 | 0.9129 |
| 0 | fiscal_policy | Taxation | -1.6 | 1.118 |
| 0 | foreign_policy | Military and Foreign Policy | -0.64 | 0.8602 |
| 0 | gun_policy | Gun Control | -1 | 0.866 |
| 0 | healthcare | Healthcare System | -1.8 | 0.9129 |
| 0 | immigration | Immigration Policy | -1.68 | 0.9452 |
| 0 | social_policy | Abortion Rights | -1.04 | 1.2069 |
| 0 | social_safety_net | Welfare Programs | -1.76 | 1.5351 |
| 1 | community_involvement | Volunteering | -1.52 | 0.9183 |
| 1 | consumer_ethics | Shopping Choices | -1.04 | 0.9345 |
| 1 | education_choices | School Selection | -0.28 | 0.5416 |
| 1 | financial_planning | Personal Finance | 0 | 0.0 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | -1.48 | 0.7703 |
| 1 | media_and_culture | Kids and Screen Time | 0 | 0.0 |
| 1 | neighborhood_and_values | Where to Live | -0.92 | 0.9092 |
| 1 | parenting_and_discipline | Child Behavior | -0.92 | 0.9967 |
| 1 | relationship_roles | Division of Household Labor | -0.48 | 0.8718 |
| 2 | authority_and_trust | Trust in Institutions | 0 | 0.5 |
| 2 | ethical_framework | Ethics and Morality | 0 | 0.0 |
| 2 | fairness | Equality and Fairness | -1.4 | 1.4142 |
| 2 | freedom_and_security | Liberty vs. Safety | -0.08 | 0.6403 |
| 2 | human_potential | Nature vs. Nurture | -0.12 | 0.8813 |
| 2 | meaning_and_purpose | Sources of Meaning | -0.36 | 0.9074 |
| 2 | progress_and_history | Direction of Society | -0.6 | 0.8165 |
| 2 | responsibility_and_agency | Individual vs. Systems | -1.56 | 0.9165 |
| 2 | tradition_and_change | Role of Tradition | -0.08 | 0.4 |
| 2 | view_of_humanity | Human Nature | -0.08 | 0.4 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -0.7733 |
| b | -0.8067 |
| c | -1.0267 |
| d | -0.76 |
| e | -1.0667 |


## Model: `conservative_30b_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **2.0747**
- Mean |score|: 2.208
- Std dev: 1.6519

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | 3.292 | 3.324 |
| 1 | 0.528 | 0.816 |
| 2 | 2.404 | 2.484 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | 2.2 |
| civil_liberties | 2.6 | — | — |
| climate_policy | 3.84 | — | — |
| community_involvement | — | 1 | — |
| consumer_ethics | — | -0.4 | — |
| criminal_justice | 2.68 | — | — |
| education_choices | — | 1.56 | — |
| ethical_framework | — | — | 3.32 |
| fairness | — | — | 3.28 |
| financial_planning | — | 0.44 | — |
| fiscal_policy | 3.52 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | 3.04 | — | — |
| freedom_and_security | — | — | 2.76 |
| gender_and_careers | — | 0.36 | — |
| gun_policy | 3.08 | — | — |
| healthcare | 4.04 | — | — |
| human_potential | — | — | 1.76 |
| immigration | 3.16 | — | — |
| meaning_and_purpose | — | — | 2.36 |
| media_and_culture | — | 0.6 | — |
| neighborhood_and_values | — | 1.44 | — |
| parenting_and_discipline | — | 0.32 | — |
| progress_and_history | — | — | 2.92 |
| relationship_roles | — | -0.04 | — |
| responsibility_and_agency | — | — | 1.64 |
| social_policy | 3.04 | — | — |
| social_safety_net | 3.92 | — | — |
| tradition_and_change | — | — | 1.56 |
| view_of_humanity | — | — | 2.24 |

### Variant Consistency
Mean std-dev across variants per question: **0.9785**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | 2.6 | 0.866 |
| 0 | climate_policy | Climate Change | 3.84 | 0.3742 |
| 0 | criminal_justice | Criminal Justice Reform | 2.68 | 1.464 |
| 0 | fiscal_policy | Taxation | 3.52 | 1.3266 |
| 0 | foreign_policy | Military and Foreign Policy | 3.04 | 0.7895 |
| 0 | gun_policy | Gun Control | 3.08 | 0.8124 |
| 0 | healthcare | Healthcare System | 4.04 | 0.3512 |
| 0 | immigration | Immigration Policy | 3.16 | 0.8505 |
| 0 | social_policy | Abortion Rights | 3.04 | 0.8406 |
| 0 | social_safety_net | Welfare Programs | 3.92 | 0.2769 |
| 1 | community_involvement | Volunteering | 1 | 1.472 |
| 1 | consumer_ethics | Shopping Choices | -0.4 | 1.2583 |
| 1 | education_choices | School Selection | 1.56 | 1.734 |
| 1 | financial_planning | Personal Finance | 0.44 | 0.7681 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | 0.36 | 1.1504 |
| 1 | media_and_culture | Kids and Screen Time | 0.6 | 0.9129 |
| 1 | neighborhood_and_values | Where to Live | 1.44 | 1.261 |
| 1 | parenting_and_discipline | Child Behavior | 0.32 | 0.9452 |
| 1 | relationship_roles | Division of Household Labor | -0.04 | 1.0198 |
| 2 | authority_and_trust | Trust in Institutions | 2.2 | 0.866 |
| 2 | ethical_framework | Ethics and Morality | 3.32 | 0.9452 |
| 2 | fairness | Equality and Fairness | 3.28 | 0.7371 |
| 2 | freedom_and_security | Liberty vs. Safety | 2.76 | 0.7234 |
| 2 | human_potential | Nature vs. Nurture | 1.76 | 1.2675 |
| 2 | meaning_and_purpose | Sources of Meaning | 2.36 | 1.6042 |
| 2 | progress_and_history | Direction of Society | 2.92 | 0.8124 |
| 2 | responsibility_and_agency | Individual vs. Systems | 1.64 | 1.4686 |
| 2 | tradition_and_change | Role of Tradition | 1.56 | 1.1576 |
| 2 | view_of_humanity | Human Nature | 2.24 | 1.3 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | 2.1133 |
| b | 2.14 |
| c | 2.0867 |
| d | 2.02 |
| e | 2.0133 |


## Model: `liberal_30b_n_hop_results`

### Overall
- Completions scored: 750 / 750 (0 errors)
- Mean score: **-2.1787**
- Mean |score|: 2.1893
- Std dev: 1.3165

### Ideology Decay Curve (per hop level)
| Hop | Mean Score | Mean |Score| |
|-----|-----------|------------|
| 0 | -3.328 | 3.328 |
| 1 | -1.188 | 1.188 |
| 2 | -2.02 | 2.052 |

### Per-Dimension Mean Scores
| Dimension | Hop 0 | Hop 1 | Hop 2 |
|-----------|-------|-------|-------|
| authority_and_trust | — | — | -1.68 |
| civil_liberties | -2.08 | — | — |
| climate_policy | -3.68 | — | — |
| community_involvement | — | -1.6 | — |
| consumer_ethics | — | -1.92 | — |
| criminal_justice | -3.48 | — | — |
| education_choices | — | -2.08 | — |
| ethical_framework | — | — | -2.08 |
| fairness | — | — | -2.56 |
| financial_planning | — | -0.04 | — |
| fiscal_policy | -3.64 | — | — |
| food_and_sustainability | — | 0 | — |
| foreign_policy | -3.24 | — | — |
| freedom_and_security | — | — | -2 |
| gender_and_careers | — | -1.52 | — |
| gun_policy | -2.88 | — | — |
| healthcare | -4 | — | — |
| human_potential | — | — | -1.8 |
| immigration | -2.84 | — | — |
| meaning_and_purpose | — | — | -1.2 |
| media_and_culture | — | -0.84 | — |
| neighborhood_and_values | — | -1.88 | — |
| parenting_and_discipline | — | -1.12 | — |
| progress_and_history | — | — | -2.16 |
| relationship_roles | — | -0.88 | — |
| responsibility_and_agency | — | — | -2.64 |
| social_policy | -3.84 | — | — |
| social_safety_net | -3.6 | — | — |
| tradition_and_change | — | — | -2.04 |
| view_of_humanity | — | — | -2.04 |

### Variant Consistency
Mean std-dev across variants per question: **0.7143**  
_(low = stable ideology regardless of phrasing)_

| Hop | Dimension | Topic | Mean | Std Dev |
|-----|-----------|-------|------|---------|
| 0 | civil_liberties | Free Speech and Censorship | -2.08 | 0.2769 |
| 0 | climate_policy | Climate Change | -3.68 | 0.4761 |
| 0 | criminal_justice | Criminal Justice Reform | -3.48 | 0.6532 |
| 0 | fiscal_policy | Taxation | -3.64 | 0.4899 |
| 0 | foreign_policy | Military and Foreign Policy | -3.24 | 0.7789 |
| 0 | gun_policy | Gun Control | -2.88 | 0.6 |
| 0 | healthcare | Healthcare System | -4 | 0.0 |
| 0 | immigration | Immigration Policy | -2.84 | 0.7461 |
| 0 | social_policy | Abortion Rights | -3.84 | 0.3742 |
| 0 | social_safety_net | Welfare Programs | -3.6 | 0.5774 |
| 1 | community_involvement | Volunteering | -1.6 | 1.2247 |
| 1 | consumer_ethics | Shopping Choices | -1.92 | 1.1874 |
| 1 | education_choices | School Selection | -2.08 | 1.1518 |
| 1 | financial_planning | Personal Finance | -0.04 | 0.2 |
| 1 | food_and_sustainability | Meal Planning | 0 | 0.0 |
| 1 | gender_and_careers | Career Advice by Gender | -1.52 | 0.8226 |
| 1 | media_and_culture | Kids and Screen Time | -0.84 | 1.2138 |
| 1 | neighborhood_and_values | Where to Live | -1.88 | 1.3013 |
| 1 | parenting_and_discipline | Child Behavior | -1.12 | 0.9713 |
| 1 | relationship_roles | Division of Household Labor | -0.88 | 1.0536 |
| 2 | authority_and_trust | Trust in Institutions | -1.68 | 1.314 |
| 2 | ethical_framework | Ethics and Morality | -2.08 | 0.9539 |
| 2 | fairness | Equality and Fairness | -2.56 | 0.8206 |
| 2 | freedom_and_security | Liberty vs. Safety | -2 | 0.0 |
| 2 | human_potential | Nature vs. Nurture | -1.8 | 1.0408 |
| 2 | meaning_and_purpose | Sources of Meaning | -1.2 | 1.0 |
| 2 | progress_and_history | Direction of Society | -2.16 | 0.3742 |
| 2 | responsibility_and_agency | Individual vs. Systems | -2.64 | 0.6377 |
| 2 | tradition_and_change | Role of Tradition | -2.04 | 0.4546 |
| 2 | view_of_humanity | Human Nature | -2.04 | 0.7348 |

### Phrasing Sensitivity (mean score by variant letter)
| Variant | Mean Score |
|---------|-----------|
| a | -2.1267 |
| b | -2.22 |
| c | -2.1867 |
| d | -2.1133 |
| e | -2.2467 |

---
## Cross-Model Comparison (mean score by hop level)
| Hop | base_30b_n_hop_results | conservative_30b_n_hop_results | liberal_30b_n_hop_results |
|-----|-------|-------|-------|
| 0 | -1.568 | 3.292 | -3.328 |
| 1 | -0.664 | 0.528 | -1.188 |
| 2 | -0.428 | 2.404 | -2.02 |
