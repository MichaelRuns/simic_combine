# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Veterinary clinical tool for predicting optimal levothyroxine (T4) dosing in hypothyroid kittens using allometric scaling. Built on clinical data from UC Davis VMTH (~344 observations across multiple kittens).

Two models available:
- **Original (controlled-only):** `Daily T4 Dose (mcg) = 142 × Weight(kg)^0.14` — 97 controlled observations, R²=0.017
- **Mixed-effects model:** `Daily T4 Dose (mcg) = 94 × Weight(kg)^0.40` (controlled) — 229 observations across 34 animals with random intercepts per animal, marginal R²=0.241, ICC=0.478

## Commands

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Run prediction CLI (prints formula, uncertainties, reference table)
python -m simic_combine.predict

# Generate all figures (PNG + PDF to figures/, interactive HTML)
python -m simic_combine.visualize
```

## Architecture

Four modules in `src/simic_combine/`, each with a single responsibility:

- **data.py** — Data pipeline: CSV loading → treatment filtering → control status filtering. Chain: `load_data()` → `get_treatment_data()` → `get_controlled_cases()`. Also: `normalize_animal_ids()` for variant name mapping, `get_mixed_model_data()` for LMM prep. Control statuses 1-5 map to Controlled, Undertreated, Overtreated, High TT4&TSH, Normal TT4 High TSH.
- **model.py** — `AllometricModel` dataclass with scipy `curve_fit()` power law fitting. `MixedAllometricModel` dataclass with statsmodels `MixedLM` for log-linearized mixed-effects model with random intercepts per animal. Both have `predict()`, `predict_per_kg()`, `predict_with_ci()` methods.
- **predict.py** — Public prediction API. Lazy-loads and caches fitted models. `predict_optimal_dose()` (original), `predict_optimal_dose_mixed()` (mixed model, supports animal-specific predictions).
- **visualize.py** — Matplotlib/Seaborn static figures (publication-quality, 300 DPI) and Plotly interactive HTML. `generate_all_figures(include_mixed=True)` for batch output including spaghetti plot, random effects plot, and model comparison.

## Key Technical Details

- Python 3.10+ with full type hints (including `@overload` in predict.py)
- Power law fitting via `scipy.optimize.curve_fit`; delta method for confidence intervals
- Original model R² is ~0.017 (weight explains little variance when treating observations as independent)
- Mixed model marginal R² is ~0.241 and ICC ~0.478 (48% of variance is between-animal, properly accounted for by random intercepts)
- Data file: `data/Hypothyroid kitten study - _Clinical Cases UTD (11_30_25).csv`
- Interactive visualization deployed via GitHub Pages (`docs/index.html`)

## Validation

After modifying models or visualizations:
- Run tests: `python -m pytest tests/`
- Regenerate figures: `python -m simic_combine.visualize`
- Regenerate GitHub Pages interactive plot: `python -c "from simic_combine.visualize import create_interactive_plot; from pathlib import Path; create_interactive_plot(output_path=Path('docs/index.html'))"`
- Verify docs/index.html reflects current model parameters

## Dependencies

Runtime: pandas, numpy, scipy, matplotlib, seaborn, plotly, statsmodels
Dev: pytest, pytest-cov
