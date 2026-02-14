"""Prediction functions for optimal thyroid dosing."""

from typing import overload

from simic_combine.data import load_data, get_treatment_data, get_controlled_cases
from simic_combine.model import fit_allometric_model, AllometricModel

# Cached model (lazy loaded)
_cached_model: AllometricModel | None = None


def get_fitted_model(force_refit: bool = False) -> AllometricModel:
    """
    Get the fitted allometric model, loading from data if needed.

    Args:
        force_refit: If True, refit model even if cached.

    Returns:
        Fitted AllometricModel instance.
    """
    global _cached_model

    if _cached_model is None or force_refit:
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        _cached_model = fit_allometric_model(
            controlled["Weight"],
            controlled["daily_dose_mcg"],
        )

    return _cached_model


@overload
def predict_optimal_dose(
    weight_kg: float,
    return_range: bool = False,
) -> float: ...


@overload
def predict_optimal_dose(
    weight_kg: float,
    return_range: bool = True,
) -> tuple[float, float, float]: ...


def predict_optimal_dose(
    weight_kg: float,
    return_range: bool = False,
) -> float | tuple[float, float, float]:
    """
    Predict optimal daily T4 dose for a hypothyroid kitten.

    Uses an allometric scaling model fitted to controlled cases from
    the clinical study. The model captures the relationship:

        Dose (mcg/day) = a × Weight(kg)^b

    where b < 1 indicates smaller kittens need higher mcg/kg doses.

    Args:
        weight_kg: Kitten's current weight in kg
        return_range: If True, return (low, optimal, high) dose range
                      representing 95% confidence interval

    Returns:
        Optimal daily dose in mcg, or (low, optimal, high) tuple if return_range=True

    Example:
        >>> predict_optimal_dose(1.0)
        82.5
        >>> predict_optimal_dose(2.0)
        136.2
        >>> predict_optimal_dose(1.0, return_range=True)
        (68.3, 82.5, 96.7)
    """
    if weight_kg <= 0:
        raise ValueError("Weight must be positive")

    model = get_fitted_model()

    if return_range:
        pred, lower, upper = model.predict_with_ci(weight_kg)
        return (float(lower[0]), float(pred[0]), float(upper[0]))
    else:
        return float(model.predict(weight_kg))


def predict_dose_per_kg(weight_kg: float) -> float:
    """
    Predict optimal dose per kg for a given weight.

    This is useful for clinicians who think in terms of mcg/kg dosing.
    The result shows how dose requirements per kg decrease with weight.

    Args:
        weight_kg: Kitten's current weight in kg

    Returns:
        Optimal dose per kg (mcg/kg/day)
    """
    if weight_kg <= 0:
        raise ValueError("Weight must be positive")

    model = get_fitted_model()
    return float(model.predict_per_kg(weight_kg))


def print_formula() -> None:
    """Print the fitted formula in clinician-friendly format."""
    model = get_fitted_model()

    print("=" * 60)
    print("OPTIMAL T4 DOSING FORMULA FOR HYPOTHYROID KITTENS")
    print("=" * 60)
    print()
    print(f"  {model.formula_string()}")
    print()
    print("Or equivalently:")
    print(f"  {model.formula_per_kg_string()}")
    print()
    print(f"Model fit: R² = {model.r_squared:.3f} (n = {model.n_observations})")
    print(f"Parameter uncertainty: a = {model.a:.1f} ± {model.a_se:.1f}, b = {model.b:.2f} ± {model.b_se:.2f}")
    print()
    print("Reference dosing table:")
    print("-" * 45)
    print(model.dose_table().to_string(index=False))
    print("=" * 60)


def get_dose_table() -> str:
    """Return a formatted dose reference table."""
    model = get_fitted_model()
    return model.dose_table().to_string(index=False)


if __name__ == "__main__":
    print_formula()
