# Hypothyroid Kitten T4 Dosing Calculator

A tool for predicting optimal levothyroxine (T4) dosing in hypothyroid kittens based on body weight.

## Clinical Summary

This calculator uses allometric scaling to predict T4 doses for congenital hypothyroid kittens. The model was developed from clinical data of kittens treated at veterinary teaching hospitals.

### Key Finding

**Smaller kittens require higher doses per kilogram than larger kittens.**

This follows the principle of metabolic scaling - smaller animals have higher metabolic rates relative to their body size and therefore require proportionally higher medication doses.

---

## Dosing Formula

```
Daily T4 Dose (mcg) = 142 × Weight(kg)^0.14
```

Or equivalently, for dose per kilogram:

```
Dose per kg (mcg/kg) = 142 × Weight(kg)^-0.86
```

---

## Quick Reference Dosing Table

| Weight (kg) | Daily Dose (mcg) | Dose per kg (mcg/kg) |
|-------------|------------------|----------------------|
| 0.5         | 129              | 259                  |
| 1.0         | 142              | 142                  |
| 1.5         | 151              | 101                  |
| 2.0         | 157              | 79                   |
| 3.0         | 166              | 55                   |
| 4.0         | 173              | 43                   |
| 5.0         | 178              | 36                   |
| 6.0         | 183              | 31                   |

**Note:** These are starting dose recommendations based on controlled cases. Individual patients may require dose adjustments based on clinical response and thyroid monitoring.

---

## Clinical Interpretation

### Control Status Definitions

When monitoring treatment response, patients are classified as:

| Status | Definition |
|--------|------------|
| **1 - Controlled** | Normal TT4, Normal TSH |
| **2 - Undertreated** | Low TT4 and/or High TSH |
| **3 - Overtreated** | High TT4, Normal TSH |
| **4 - Elevated TT4 & TSH** | Both TT4 and TSH elevated |
| **5 - Normal TT4, High TSH** | TT4 normal but TSH elevated |

The dosing formula was derived from observations where patients achieved **Control Status 1** (biochemically controlled).

---

## Understanding the Model

### Why do smaller kittens need more mcg/kg?

This is consistent with allometric scaling principles seen across species and drug classes:

1. **Higher metabolic rate** - Smaller animals have faster metabolism relative to body size
2. **Faster drug clearance** - Medications are metabolized more quickly in smaller patients
3. **Surface area to volume ratio** - Smaller bodies lose heat faster and have proportionally higher energy demands

### Model Limitations

- **R² = 0.017** - There is substantial individual variation in dose requirements
- The model provides a **starting point** for dosing, not a definitive prescription
- Clinical judgment and monitoring remain essential
- Some patients may require doses outside the predicted range

---

## Figures

The `figures/` directory contains publication-quality visualizations:

| Figure | Description |
|--------|-------------|
| `fig1_dose_vs_weight` | Total daily dose vs body weight |
| `fig2_dose_per_kg_vs_weight` | Dose per kg vs weight (key clinical insight) |
| `fig3_dose_per_kg_vs_age` | How dose requirements change with age |
| `fig4_individual_trajectories` | Individual patient dosing journeys |
| `fig5_control_distribution` | Control status outcomes by dose range |
| `summary_figure` | Combined 4-panel overview |

---

## Running the Calculator

### Prerequisites

- Python 3.10 or higher
- Required packages: pandas, numpy, scipy, matplotlib, seaborn

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e .
```

### Usage

**Print the dosing formula and reference table:**
```bash
python -m simic_combine.predict
```

**Generate all figures:**
```bash
python -m simic_combine.visualize
```

**Use in Python:**
```python
from simic_combine import predict_optimal_dose

# Predict dose for a 2 kg kitten
dose = predict_optimal_dose(2.0)
print(f"Recommended daily dose: {dose:.0f} mcg")
# Output: Recommended daily dose: 157 mcg

# Get dose range (95% confidence interval)
low, optimal, high = predict_optimal_dose(2.0, return_range=True)
print(f"Dose range: {low:.0f} - {high:.0f} mcg")
```

---

## Data Source

Clinical data from hypothyroid kittens treated at UC Davis VMTH and collaborating institutions. The dataset includes longitudinal observations across multiple patients with varying treatment protocols and outcomes.

---

## Contact

For questions about this tool or the underlying clinical study, please contact the research team.

---

*This tool is intended for veterinary professional use. Dosing recommendations should be used in conjunction with clinical judgment and appropriate patient monitoring.*
