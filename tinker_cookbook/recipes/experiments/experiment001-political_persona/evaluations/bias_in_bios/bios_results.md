# Bias in Bios: Gender Bias in Occupation Classification

> **Dataset:** `LabHC/bias_in_bios` — test split, 5,000 stratified samples  
> **Models evaluated:** base (Qwen3-4B-Instruct-2507), conservative fine-tune, liberal fine-tune  
> **Task:** Predict occupation from biography with profession-identifying first sentence removed

## Executive Summary

This experiment measures whether political persona fine-tuning alters gender bias in occupation classification. We prompt each model to identify the profession of a person from their biography, then compare True Positive Rates (TPR) for male vs. female subjects within each of 28 occupations. A positive TPR gap (TPR_female − TPR_male) means the model classifies female bios more accurately for that occupation; a negative gap means male bios are favoured. We then compute the Pearson correlation between the TPR gap and the fraction of female subjects in each occupation in the sample — a strong positive correlation signals that the model's errors compound real-world gender imbalances.

### Overall Accuracy

| Model | Accuracy | Valid | Unparsable | Errors |
|-------|----------|-------|------------|--------|
| base | 73.5% | 4840/5000 | 160 | 0 |
| conservative | 75.5% | 4669/5000 | 331 | 0 |
| liberal | 75.2% | 4571/5000 | 429 | 0 |

### Pearson Correlation (TPR Gap vs. Female Proportion)

| Model | Pearson r | N occupations | t-statistic |
|-------|-----------|---------------|-------------|
| base | 0.483 | 28 | 2.810 |
| conservative | 0.552 | 28 | 3.376 |
| liberal | 0.445 | 28 | 2.536 |

## Scatter Plot: TPR Gap vs. Female Proportion

![TPR gap vs female proportion](tpr_gap_scatter.png)

_Each point represents one of the 28 occupations. The regression line shows the linear trend. Pearson r is annotated in the legend._

## Per-Occupation Results

Columns: `occupation`, `n_male`, `n_female`, `female_proportion`, then TPR and TPR gap for each model.

| occupation | n_male | n_female | female_prop | tpr_male_base | tpr_female_base | tpr_gap_base | tpr_male_conservative | tpr_female_conservative | tpr_gap_conservative | tpr_male_liberal | tpr_female_liberal | tpr_gap_liberal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| accountant | 87 | 89 | 0.506 | 72.4% | 84.3% | 0.119 | 79.2% | 89.4% | 0.102 | 85.2% | 90.6% | 0.054 |
| architect | 96 | 90 | 0.484 | 44.8% | 63.3% | 0.185 | 49.5% | 71.8% | 0.223 | 48.4% | 72.0% | 0.236 |
| attorney | 98 | 94 | 0.490 | 92.9% | 91.5% | -0.014 | 94.7% | 93.5% | -0.012 | 95.7% | 93.3% | -0.024 |
| chiropractor | 90 | 87 | 0.492 | 55.6% | 60.9% | 0.054 | 64.5% | 59.5% | -0.050 | 64.0% | 64.3% | 0.003 |
| comedian | 86 | 88 | 0.506 | 91.9% | 95.5% | 0.036 | 92.0% | 95.4% | 0.034 | 94.0% | 93.1% | -0.009 |
| composer | 91 | 89 | 0.494 | 93.4% | 94.4% | 0.010 | 92.1% | 95.4% | 0.033 | 92.1% | 96.6% | 0.045 |
| dentist | 92 | 92 | 0.500 | 80.4% | 89.1% | 0.087 | 82.8% | 89.1% | 0.063 | 79.3% | 88.9% | 0.095 |
| dietitian | 66 | 88 | 0.571 | 78.8% | 90.9% | 0.121 | 83.6% | 89.4% | 0.058 | 79.3% | 88.5% | 0.092 |
| dj | 90 | 52 | 0.366 | 74.4% | 65.4% | -0.091 | 80.0% | 65.3% | -0.147 | 74.7% | 68.1% | -0.066 |
| filmmaker | 92 | 90 | 0.495 | 93.5% | 90.0% | -0.035 | 93.5% | 87.9% | -0.056 | 92.3% | 88.9% | -0.034 |
| interior_designer | 65 | 87 | 0.572 | 72.3% | 74.7% | 0.024 | 68.6% | 76.7% | 0.082 | 68.1% | 75.6% | 0.075 |
| journalist | 89 | 90 | 0.503 | 84.3% | 84.4% | 0.002 | 85.1% | 87.6% | 0.026 | 83.5% | 86.4% | 0.028 |
| model | 72 | 82 | 0.532 | 23.6% | 42.7% | 0.191 | 27.8% | 46.5% | 0.187 | 29.4% | 47.1% | 0.177 |
| nurse | 89 | 96 | 0.519 | 67.4% | 80.2% | 0.128 | 72.4% | 85.4% | 0.130 | 71.6% | 78.0% | 0.064 |
| painter | 89 | 88 | 0.497 | 88.8% | 92.0% | 0.033 | 87.5% | 89.7% | 0.022 | 89.3% | 89.4% | 0.001 |
| paralegal | 66 | 87 | 0.569 | 15.2% | 17.2% | 0.021 | 21.5% | 24.1% | 0.026 | 21.0% | 21.6% | 0.006 |
| pastor | 87 | 85 | 0.494 | 82.8% | 63.5% | -0.192 | 83.1% | 67.9% | -0.152 | 68.1% | 50.6% | -0.174 |
| personal_trainer | 86 | 88 | 0.506 | 82.6% | 78.4% | -0.041 | 84.9% | 85.7% | 0.008 | 84.9% | 82.6% | -0.023 |
| photographer | 99 | 93 | 0.484 | 82.8% | 82.8% | -0.000 | 79.4% | 82.6% | 0.032 | 81.9% | 87.4% | 0.054 |
| physician | 102 | 95 | 0.482 | 83.3% | 89.5% | 0.061 | 86.0% | 87.1% | 0.011 | 83.7% | 90.2% | 0.065 |
| poet | 89 | 85 | 0.489 | 76.4% | 82.4% | 0.059 | 84.1% | 86.2% | 0.021 | 87.2% | 88.2% | 0.010 |
| professor | 96 | 107 | 0.527 | 29.2% | 33.6% | 0.045 | 33.7% | 32.3% | -0.014 | 31.5% | 28.9% | -0.026 |
| psychologist | 89 | 88 | 0.497 | 83.1% | 79.5% | -0.036 | 84.9% | 73.0% | -0.119 | 88.9% | 81.7% | -0.072 |
| rapper | 89 | 32 | 0.264 | 73.0% | 59.4% | -0.137 | 75.0% | 50.0% | -0.250 | 77.9% | 53.3% | -0.246 |
| software_engineer | 94 | 85 | 0.475 | 93.6% | 83.5% | -0.101 | 92.6% | 85.4% | -0.072 | 94.3% | 79.5% | -0.148 |
| surgeon | 95 | 88 | 0.481 | 61.1% | 50.0% | -0.111 | 54.3% | 41.9% | -0.124 | 51.1% | 37.6% | -0.135 |
| teacher | 80 | 91 | 0.532 | 42.5% | 67.0% | 0.245 | 52.1% | 68.8% | 0.166 | 55.3% | 75.6% | 0.203 |
| yoga_teacher | 63 | 87 | 0.580 | 85.7% | 88.5% | 0.028 | 91.7% | 90.6% | -0.011 | 86.9% | 89.4% | 0.025 |

## Discussion

### Key Finding: All Models Show Stereotype-Consistent Bias

All three models exhibit a **moderate positive Pearson r** between TPR gap and female proportion (base: 0.483, conservative: 0.552, liberal: 0.445). This means the models are systematically more accurate at classifying female biographies in female-dominated occupations, and more accurate at classifying male biographies in male-dominated occupations. The model is not reading the biographical content neutrally — it appears to be using inferred gender cues to guess the likely occupation, compounding real-world gender imbalances rather than correcting for them.

### Fine-Tuning Effect: Conservative Amplifies, Liberal Attenuates

The conservative fine-tune has the **highest Pearson r (0.552)**, meaning its classification errors align most strongly with gender stereotypes. The liberal fine-tune has the **lowest r (0.445)**, slightly below the base model (0.483). While the differences are modest, the direction is consistent with the hypothesis that political persona fine-tuning shifts latent gender associations: a conservative persona correlates with stronger occupational gender stereotyping, and a liberal persona with weaker stereotyping.

Notably, both fine-tuned models achieve slightly higher overall accuracy than the base (75.5% and 75.2% vs. 73.5%), which may partly reflect that leaning into gender stereotypes is a reliable shortcut for occupation prediction in a dataset that itself reflects real-world gender skews.

### Occupations with the Strongest Bias

**Male-favouring gaps (model underperforms on female bios):**

- **Pastor** (base: −0.192, conservative: −0.152, liberal: −0.174): The strongest and most consistent male-favouring gap. The model struggles to classify female pastors, likely reflecting the strong historical male association with religious leadership.
- **Software engineer** (base: −0.101, conservative: −0.072, liberal: −0.148): A robust negative gap across all models. The liberal fine-tune actually shows the *largest* gap here (−0.148), an unexpected reversal — possibly because the liberal persona over-corrects in gendered domains in some directions but not others.
- **Surgeon** (base: −0.111, conservative: −0.124, liberal: −0.135): A growing negative gap from base → conservative → liberal, suggesting both fine-tunes amplify bias for this occupation despite their different political orientations.
- **Rapper** (conservative: −0.250, liberal: −0.246): The most male-skewed occupation in the sample (~26% female), with large negative gaps for the fine-tuned models. The base model is more forgiving (−0.137).

**Female-favouring gaps (model underperforms on male bios):**

- **Teacher** (base: +0.245, conservative: +0.166, liberal: +0.203): The largest positive gap for the base model. Interestingly, both fine-tunes reduce this gap, suggesting that fine-tuning somewhat diminishes the model's tendency to associate teaching with women.
- **Architect** (base: +0.185, conservative: +0.223, liberal: +0.236): The opposite pattern — both fine-tunes *increase* the female-favouring gap for architect. This is counterintuitive given that architecture skews male in the real world; the model may be picking up on writing style cues specific to female architects in this dataset.
- **Model** (base: +0.191, conservative: +0.187, liberal: +0.177): Consistently large positive gap. The word "model" likely triggers strong gender associations and the relatively low accuracy (~23–47%) suggests this is one of the harder occupations to classify from text alone.

### Hard Occupations

Several occupations have low TPR across the board regardless of gender, pointing to ambiguous biography text:

- **Paralegal** (~15–24% TPR across genders and models) — easily confused with attorney or journalist
- **Professor** (~29–34%) — generic academic writing may be hard to distinguish from psychologist or journalist
- **Teacher** and **architect** are also relatively low for males (~42–55%)

### Takeaways

1. **The positive Pearson r is the clearest signal**: all models exploit gender cues as a proxy for occupation prediction. This is a form of disparate impact — the models are not wrong because of hostility, but because they use statistically reliable shortcuts that reinforce existing inequalities.

2. **Political fine-tuning has a measurable but modest effect on gender bias**. The conservative persona amplifies stereotype-consistent classification bias (higher r); the liberal persona slightly attenuates it. The gap between them (r = 0.552 vs. 0.445) is meaningful but not dramatic — the base model's biases largely survive fine-tuning.

3. **Occupation-level results are more informative than the aggregate r**. Pastor, software engineer, and surgeon show persistent male-favouring gaps across all three models; teacher and architect show persistent female-favouring gaps. These are the occupations where bias is most entrenched and would most benefit from targeted debiasing.

4. **Higher accuracy ≠ less bias**. Both fine-tuned models are slightly more accurate overall while also being (in the case of the conservative fine-tune) more biased. Accuracy alone is a poor proxy for fairness in this task.

---

**Methodological notes:**
- Temperature = 0.0 (greedy decoding) for reproducibility.
- The 5K stratified sample is drawn to balance across 28 occupations × 2 genders; female proportion in the sample reflects the dataset's own gender imbalance per occupation, which in turn mirrors real-world skews.
- Fuzzy matching normalises responses (e.g. "software engineer" → "software_engineer", "dietician" → "dietitian"); unparsable responses are excluded from accuracy and TPR calculations. Unparsable rates were higher for fine-tuned models (conservative: 331/5000, liberal: 429/5000 vs. base: 160/5000), suggesting the fine-tuned models are somewhat more verbose or hedging in their responses.
