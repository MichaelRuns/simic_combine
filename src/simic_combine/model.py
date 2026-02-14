"""Allometric model fitting for thyroid dosing prediction."""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd


@dataclass
class AllometricModel:
    """Fitted allometric model parameters and statistics."""

    # Primary parameters: Dose = a * Weight^b
    a: float  # Scaling coefficient
    b: float  # Allometric exponent

    # Parameter uncertainties (standard errors)
    a_se: float
    b_se: float

    # Goodness of fit
    r_squared: float
    n_observations: int

    # Raw covariance matrix for confidence intervals
    covariance: np.ndarray

    def predict(self, weight: float | np.ndarray) -> float | np.ndarray:
        """Predict optimal daily dose for given weight(s)."""
        return self.a * np.power(weight, self.b)

    def predict_per_kg(self, weight: float | np.ndarray) -> float | np.ndarray:
        """Predict optimal dose per kg for given weight(s)."""
        return self.a * np.power(weight, self.b - 1)

    def predict_with_ci(
        self, weight: float | np.ndarray, confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict dose with confidence interval.

        Args:
            weight: Weight(s) in kg
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (prediction, lower_bound, upper_bound)
        """
        weight = np.atleast_1d(weight)
        pred = self.predict(weight)

        # Delta method for confidence interval
        # For f(a,b) = a * W^b, partial derivatives are:
        # df/da = W^b
        # df/db = a * W^b * ln(W)
        log_w = np.log(weight)

        # Jacobian at each weight point
        var_pred = np.zeros_like(weight, dtype=float)
        for i, w in enumerate(weight):
            grad = np.array([w**self.b, self.a * (w**self.b) * np.log(w)])
            var_pred[i] = grad @ self.covariance @ grad

        se_pred = np.sqrt(var_pred)

        # t-value for confidence interval
        alpha = 1 - confidence
        t_val = stats.t.ppf(1 - alpha / 2, self.n_observations - 2)

        lower = pred - t_val * se_pred
        upper = pred + t_val * se_pred

        return pred, lower, upper

    def formula_string(self) -> str:
        """Return human-readable formula string."""
        return f"Dose (mcg/day) = {self.a:.1f} × Weight(kg)^{self.b:.2f}"

    def formula_per_kg_string(self) -> str:
        """Return human-readable formula for dose per kg."""
        return f"Dose (mcg/kg) = {self.a:.1f} × Weight(kg)^{self.b - 1:.2f}"

    def dose_table(self, weights: list[float] | None = None) -> pd.DataFrame:
        """
        Generate a dose reference table for common weights.

        Args:
            weights: List of weights to include. Defaults to [0.5, 1, 1.5, 2, 3, 4, 5, 6].

        Returns:
            DataFrame with weight, predicted dose, and dose per kg.
        """
        if weights is None:
            weights = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

        doses = self.predict(np.array(weights))
        dose_per_kg = self.predict_per_kg(np.array(weights))

        return pd.DataFrame({
            "Weight (kg)": weights,
            "Daily Dose (mcg)": np.round(doses, 1),
            "Dose per kg (mcg/kg)": np.round(dose_per_kg, 1),
        })


def power_law(weight: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law function: Dose = a * Weight^b"""
    return a * np.power(weight, b)


def fit_allometric_model(
    weights: np.ndarray | pd.Series,
    daily_doses: np.ndarray | pd.Series,
    initial_guess: tuple[float, float] = (50.0, 0.75),
) -> AllometricModel:
    """
    Fit allometric power law model to dose-weight data.

    The model fits: Dose = a × Weight^b

    For metabolic scaling, we expect b ≈ 0.75 (Kleiber's law).
    Values of b < 1 indicate smaller animals need more mcg/kg.

    Args:
        weights: Array of kitten weights in kg
        daily_doses: Array of total daily T4 doses in mcg
        initial_guess: Starting values for (a, b) optimization

    Returns:
        Fitted AllometricModel with parameters and statistics.
    """
    weights = np.asarray(weights)
    daily_doses = np.asarray(daily_doses)

    # Remove any NaN/inf values
    valid = np.isfinite(weights) & np.isfinite(daily_doses) & (weights > 0) & (daily_doses > 0)
    weights = weights[valid]
    daily_doses = daily_doses[valid]

    # Fit the power law
    params, covariance = curve_fit(
        power_law,
        weights,
        daily_doses,
        p0=initial_guess,
        bounds=([0, 0], [np.inf, 2]),  # a > 0, 0 < b < 2
        maxfev=5000,
    )

    a, b = params
    a_se, b_se = np.sqrt(np.diag(covariance))

    # Calculate R-squared
    predicted = power_law(weights, a, b)
    ss_res = np.sum((daily_doses - predicted) ** 2)
    ss_tot = np.sum((daily_doses - np.mean(daily_doses)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return AllometricModel(
        a=a,
        b=b,
        a_se=a_se,
        b_se=b_se,
        r_squared=r_squared,
        n_observations=len(weights),
        covariance=covariance,
    )


def compare_control_groups(df: pd.DataFrame) -> dict[int, AllometricModel]:
    """
    Fit separate models to each control status group.

    This helps visualize how dose-weight relationships differ
    between controlled, undertreated, and overtreated cases.

    Args:
        df: Treatment dataframe with 'Weight', 'daily_dose_mcg', 'Control' columns

    Returns:
        Dictionary mapping control status to fitted model.
    """
    models = {}

    for control_status in sorted(df["Control"].unique()):
        subset = df[df["Control"] == control_status]
        if len(subset) >= 5:  # Need minimum observations for fitting
            try:
                model = fit_allometric_model(
                    subset["Weight"],
                    subset["daily_dose_mcg"],
                )
                models[control_status] = model
            except RuntimeError:
                # Curve fit failed, skip this group
                pass

    return models
