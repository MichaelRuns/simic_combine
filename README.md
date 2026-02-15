# Hypothyroid Kitten T4 Dosing Calculator

A tool for predicting optimal levothyroxine (T4) dosing in hypothyroid kittens based on body weight, using allometric scaling models developed from clinical data at UC Davis VMTH.

## Clinical Summary

**Smaller kittens require higher doses per kilogram than larger kittens.** This follows allometric scaling principles: smaller animals have higher metabolic rates relative to body size and clear medications faster, requiring proportionally higher doses.

Two models are available: a simple power-law fit to controlled cases only, and a mixed-effects model that accounts for repeated measurements within animals across all treatment observations.

---

## The Data

The dataset contains 343 raw observations from hypothyroid kittens treated at UC Davis VMTH and collaborating institutions. After filtering to treatment observations (T4 > 0, valid weight and control status), 229 observations remain across 34 unique animals. Each animal was measured between 1 and 19 times over the course of treatment, creating a longitudinal dataset.

Each visit is classified into one of five control status categories:

| Status | Definition |
|--------|------------|
| **1 - Controlled** | Normal TT4, Normal TSH |
| **2 - Undertreated** | Low TT4 and/or High TSH |
| **3 - Overtreated** | High TT4, Normal TSH |
| **4 - Elevated TT4 & TSH** | Both TT4 and TSH elevated |
| **5 - Normal TT4, High TSH** | TT4 normal but TSH elevated |

Of the 229 treatment observations, 97 achieved control status 1 (biochemically controlled). The remaining 132 represent undertreated, overtreated, or partially controlled cases. This longitudinal structure — the same kitten measured repeatedly over time — is central to choosing the right statistical model.

---

## Dosing Formulas

### Model 1 — Simple Allometric Scaling (Original)

```
Daily T4 Dose (mcg) = 142 × Weight(kg)^0.14
```

Fitted via nonlinear least squares (`scipy.optimize.curve_fit`) to the 97 controlled-only observations. Each observation is treated as independent.

- **n = 97** controlled cases
- **R² = 0.017** — weight explains only 1.7% of dose variance
- Limitation: discards 132 non-controlled observations, and treats repeated measurements from the same kitten as independent data points

### Model 2 — Mixed-Effects Model (Recommended)

```
Daily T4 Dose (mcg) = 94 × Weight(kg)^0.40    (controlled, population-level)
```

A linear mixed-effects model fitted to all 229 treatment observations using `statsmodels.MixedLM`. The approach:

1. **Log-linearization** — The power law `Dose = a × Weight^b` becomes a linear model on the log scale: `log(Dose) = log(a) + b × log(Weight)`. This allows standard mixed-effects machinery.
2. **Random intercepts per animal** — Each of the 34 kittens gets its own baseline dose level. Some kittens consistently need more medication than others, and the model captures this.
3. **Control status covariate** — An `is_controlled` indicator uses all 229 observations while adjusting for whether each visit achieved biochemical control. The population prediction above is for controlled cases.

Key statistics:
- **n = 229** observations across **34 animals**
- **ICC = 0.478** — 48% of the residual variance is between-animal (knowing *which kitten* explains almost half the remaining variation)
- **Marginal R² = 0.241** — fixed effects (weight + control status) explain 24% of variance
- **Allometric exponent b = 0.40** (vs. 0.14 in the original model)

---

## Quick Reference Dosing Table

Mixed-effects model predictions (primary) with original model for comparison:

| Weight (kg) | Mixed Model (mcg) | Mixed (mcg/kg) | Original (mcg) | Original (mcg/kg) |
|-------------|-------------------|----------------|-----------------|-------------------|
| 1.0         | 94                | 94             | 142             | 142               |
| 1.5         | 111               | 74             | 151             | 100               |
| 2.0         | 124               | 62             | 157             | 78                |
| 3.0         | 147               | 49             | 166             | 55                |
| 4.0         | 165               | 41             | 173             | 43                |
| 5.0         | 180               | 36             | 178             | 36                |
| 6.0         | 194               | 32             | 183             | 30                |

The models converge around 5 kg and diverge most for small kittens, where the mixed model predicts substantially lower doses. This is because the mixed model's steeper allometric exponent (b=0.40 vs. b=0.14) produces a dose curve that rises more steeply with weight — meaning it assigns less dose to small kittens and more to large ones.

**Note:** These are starting dose recommendations. Individual patients may require dose adjustments based on clinical response and thyroid monitoring. There is substantial individual variation in dose requirements.

---

## Understanding the Models

### Why do smaller kittens need more mcg/kg?

This is consistent with allometric scaling principles seen across species and drug classes:

1. **Higher metabolic rate** — Smaller animals have faster metabolism relative to body size
2. **Faster drug clearance** — Medications are metabolized more quickly in smaller patients
3. **Surface area to volume ratio** — Smaller bodies lose heat faster and have proportionally higher energy demands

### The repeated measures problem

The original model treats each visit as an independent observation, but the same kitten measured 5-19 times means these observations are correlated. A kitten that needs a high dose at visit 1 will likely need a high dose at visit 5. Standard regression ignores this correlation, which:

- **Underestimates uncertainty** — pseudo-replication inflates the apparent sample size
- **Lets high-visit kittens dominate** — a kitten measured 19 times has 19× the influence of one measured once
- **Mixes within-animal and between-animal variation** — dose changes within a growing kitten are different from dose differences between kittens

### How the mixed model solves this

The mixed-effects model separates the signal into:

- **Fixed effects** (population-level): the weight-dose relationship and the control status adjustment that apply to all kittens
- **Random intercepts** (animal-level): each kitten's individual baseline, capturing that some kittens consistently need more or less medication than average

The ICC of 0.478 means that once you account for weight and control status, knowing *which kitten* you're treating explains 48% of the remaining variance. This is a strong animal-level effect.

### Why R² improved from 0.017 to 0.241

The original model's low R² partly reflects treating correlated within-animal observations as independent noise. The mixed model:

- Uses all 229 observations (vs. 97), increasing statistical power
- Separates within-animal from between-animal variance, estimating the weight effect more precisely
- Includes control status as a covariate, explaining additional variance
- Estimates a steeper allometric exponent (b=0.40 vs. b=0.14), better capturing the weight-dose relationship

### Model Validation

The original allometric model was validated against empirical medians from controlled cases:

| Weight | Model Prediction | Actual Median | Difference |
|--------|------------------|---------------|------------|
| 1.5 kg | 100 mcg/kg       | 100 mcg/kg    | 0          |
| 2.0 kg | 78 mcg/kg        | 71 mcg/kg     | +7         |
| 3.0 kg | 55 mcg/kg        | 56 mcg/kg     | 0          |
| 4.0 kg | 43 mcg/kg        | 37 mcg/kg     | +7         |
| 5.0 kg | 36 mcg/kg        | 31 mcg/kg     | +5         |

### Limitations

- **Individual variation remains high** — even the mixed model's marginal R²=0.241 means 76% of variance is unexplained by weight and control status alone (though 48% of the residual is captured by knowing which animal)
- **Limited data below 1.5 kg** — only 4 controlled cases between 1.0-1.5 kg; predictions for the smallest kittens are extrapolations
- Both models provide a **starting point** for dosing, not a definitive prescription
- Clinical judgment and monitoring remain essential

---

## Figures

The `figures/` directory contains publication-quality visualizations (300 DPI PDF + PNG):

| Figure | Description |
|--------|-------------|
| `fig1_dose_vs_weight` | Total daily dose vs body weight, colored by control status, with original model curve |
| `fig2_dose_per_kg_vs_weight` | Dose per kg vs weight — key clinical insight showing smaller kittens need higher mcg/kg |
| `fig3_dose_per_kg_vs_age` | Dose requirements over time (age in days) |
| `fig4_individual_trajectories` | Individual kitten dose trajectories over time, controlled visits highlighted with markers |
| `fig5_control_distribution` | Stacked bar: control status distribution by dose range, shows the dosing "sweet spot" |
| `fig6_spaghetti_plot` | Each animal's dose-weight trajectory as connected lines with mixed model population curve overlaid — visually demonstrates within-animal correlation |
| `fig7_random_effects` | Forest plot of per-animal random intercepts showing which animals need consistently higher/lower doses than average, with ICC label |
| `fig8_model_comparison` | Original vs mixed-effects model curves overlaid on controlled data — the mixed model has a steeper curve (b=0.40 vs b=0.14) |
| `summary_figure` | Combined 4-panel overview |

An interactive dosing calculator is also available at `docs/index.html` (deployed via GitHub Pages).

---

## Running the Calculator

### Prerequisites

- Python 3.10 or higher
- Required packages: pandas, numpy, scipy, matplotlib, seaborn, plotly, statsmodels

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

### Usage

**Print dosing formulas and reference tables (both models):**
```bash
python -m simic_combine.predict
```

**Generate all figures:**
```bash
python -m simic_combine.visualize
```

**Use in Python — original model:**
```python
from simic_combine import predict_optimal_dose

# Predict dose for a 2 kg kitten
dose = predict_optimal_dose(2.0)
print(f"Recommended daily dose: {dose:.0f} mcg")

# Get dose range (95% confidence interval)
low, optimal, high = predict_optimal_dose(2.0, return_range=True)
print(f"Dose range: {low:.0f} - {high:.0f} mcg")
```

**Use in Python — mixed-effects model (recommended):**
```python
from simic_combine import predict_optimal_dose_mixed

# Population-level prediction for a 2 kg kitten
dose = predict_optimal_dose_mixed(2.0)
print(f"Recommended daily dose: {dose:.0f} mcg")

# Get dose range
low, optimal, high = predict_optimal_dose_mixed(2.0, return_range=True)
print(f"Dose range: {low:.0f} - {high:.0f} mcg")

# Animal-specific prediction (if you know the kitten)
dose = predict_optimal_dose_mixed(2.0, animal_id="Figaro")
```

---

## Data Source

Clinical data from hypothyroid kittens treated at UC Davis VMTH and collaborating institutions. The dataset includes longitudinal observations across multiple patients with varying treatment protocols and outcomes.

---

## Contact

For questions about this tool or the underlying clinical study, please contact the research team.

---

*This tool is intended for veterinary professional use. Dosing recommendations should be used in conjunction with clinical judgment and appropriate patient monitoring.*
