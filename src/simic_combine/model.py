"""Allometric model fitting for thyroid dosing prediction."""

from dataclasses import dataclass, field
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
            weights: List of weights to include. Defaults to [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0].
                     Note: Starts at 1.0 kg as minimum controlled case in data is 1.25 kg.

        Returns:
            DataFrame with weight, predicted dose, and dose per kg.
        """
        if weights is None:
            # Start at 1.0 kg - minimum controlled case is 1.25 kg
            # Values below 1.0 kg would be extrapolation outside observed data
            weights = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

        doses = self.predict(np.array(weights))
        dose_per_kg = self.predict_per_kg(np.array(weights))

        return pd.DataFrame({
            "Weight (kg)": weights,
            "Daily Dose (mcg)": np.round(doses, 0).astype(int),
            "Dose per kg (mcg/kg)": np.round(dose_per_kg, 0).astype(int),
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


@dataclass
class MixedAllometricModel:
    """Fitted mixed-effects allometric model with random intercepts per animal.

    The model is fit on the log scale:
        log(dose) = log(a) + b * log(weight) + c * is_controlled + (1 | Animal_ID)

    Population prediction for controlled cases:
        dose = a_controlled * weight^b
    where a_controlled = exp(log_a + control_effect)
    """

    # Population parameters (back-transformed to original scale)
    a: float  # Scaling coefficient (for non-controlled baseline)
    b: float  # Allometric exponent
    a_se: float
    b_se: float

    # Control effect (on log scale)
    control_effect: float
    control_effect_se: float

    # Derived: population coefficient for controlled cases
    a_controlled: float

    # Random effects
    n_animals: int
    random_intercept_sd: float
    residual_sd: float
    icc: float  # Intraclass correlation coefficient
    animal_random_effects: dict[str, float] = field(default_factory=dict)

    # Fit statistics
    n_observations: int = 0
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    r_squared_marginal: float = 0.0

    # Log-scale parameters (for CI computation)
    log_a: float = 0.0
    log_a_se: float = 0.0
    _fe_cov: np.ndarray = field(default_factory=lambda: np.array([]))

    def predict(self, weight: float | np.ndarray, controlled: bool = True) -> float | np.ndarray:
        """Predict population-level daily dose for given weight(s).

        Args:
            weight: Weight(s) in kg.
            controlled: If True, predict for controlled cases (default).
        """
        coef = self.a_controlled if controlled else self.a
        return coef * np.power(weight, self.b)

    def predict_per_kg(self, weight: float | np.ndarray, controlled: bool = True) -> float | np.ndarray:
        """Predict population-level dose per kg for given weight(s)."""
        coef = self.a_controlled if controlled else self.a
        return coef * np.power(weight, self.b - 1)

    def predict_animal(self, weight: float | np.ndarray, animal_id: str, controlled: bool = True) -> float | np.ndarray:
        """Predict dose for a specific animal using its random intercept."""
        re = self.animal_random_effects.get(animal_id, 0.0)
        log_a_base = self.log_a + re
        if controlled:
            log_a_base += self.control_effect
        return np.exp(log_a_base) * np.power(weight, self.b)

    def predict_with_ci(
        self, weight: float | np.ndarray, confidence: float = 0.95, controlled: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict dose with confidence interval (population-level, delta method on log scale)."""
        weight = np.atleast_1d(weight)
        log_w = np.log(weight)

        # Mean prediction on log scale
        log_pred = self.log_a + self.b * log_w
        if controlled:
            log_pred = log_pred + self.control_effect

        # Variance via delta method using fixed-effects covariance
        # Gradient: [d/d(log_a), d/d(b), d/d(control_effect)] = [1, log_w, is_controlled]
        alpha = 1 - confidence
        z_val = stats.norm.ppf(1 - alpha / 2)

        pred = np.exp(log_pred)
        lower = np.zeros_like(weight, dtype=float)
        upper = np.zeros_like(weight, dtype=float)

        if self._fe_cov.size > 0:
            for i, lw in enumerate(log_w):
                if controlled:
                    grad = np.array([1.0, lw, 1.0])
                else:
                    grad = np.array([1.0, lw, 0.0])
                var_log = grad @ self._fe_cov @ grad
                se_log = np.sqrt(var_log)
                lower[i] = np.exp(log_pred[i] - z_val * se_log)
                upper[i] = np.exp(log_pred[i] + z_val * se_log)
        else:
            # Fallback: use parameter SEs
            se_log = self.log_a_se
            lower = np.exp(log_pred - z_val * se_log)
            upper = np.exp(log_pred + z_val * se_log)

        return pred, lower, upper

    def formula_string(self, controlled: bool = True) -> str:
        """Return human-readable formula string."""
        coef = self.a_controlled if controlled else self.a
        label = " (controlled)" if controlled else " (baseline)"
        return f"Dose (mcg/day) = {coef:.1f} × Weight(kg)^{self.b:.2f}{label}"

    def dose_table(self, weights: list[float] | None = None, controlled: bool = True) -> pd.DataFrame:
        """Generate a dose reference table for common weights."""
        if weights is None:
            weights = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]

        doses = self.predict(np.array(weights), controlled=controlled)
        dose_per_kg = self.predict_per_kg(np.array(weights), controlled=controlled)

        return pd.DataFrame({
            "Weight (kg)": weights,
            "Daily Dose (mcg)": np.round(doses, 0).astype(int),
            "Dose per kg (mcg/kg)": np.round(dose_per_kg, 0).astype(int),
        })


def fit_mixed_model(df: pd.DataFrame) -> MixedAllometricModel:
    """
    Fit a linear mixed-effects model on log-transformed dose-weight data.

    Model: log(dose) = log(a) + b * log(weight) + c * is_controlled + (1 | Animal_ID)

    Args:
        df: DataFrame from get_mixed_model_data() with columns:
            log_dose, log_weight, is_controlled, Animal_ID

    Returns:
        Fitted MixedAllometricModel instance.
    """
    import statsmodels.formula.api as smf

    # Fit mixed model
    model = smf.mixedlm(
        "log_dose ~ log_weight + is_controlled",
        data=df,
        groups=df["Animal_ID"],
    )
    result = model.fit(reml=True)

    # Extract fixed effects
    log_a = result.fe_params["Intercept"]
    b = result.fe_params["log_weight"]
    control_effect = result.fe_params["is_controlled"]

    # Standard errors
    log_a_se = result.bse_fe["Intercept"]
    b_se = result.bse_fe["log_weight"]
    control_effect_se = result.bse_fe["is_controlled"]

    # Back-transform to original scale
    a = np.exp(log_a)
    # SE of a via delta method: se(exp(x)) = exp(x) * se(x)
    a_se = a * log_a_se
    a_controlled = np.exp(log_a + control_effect)

    # Random effects
    random_intercept_var = float(result.cov_re.iloc[0, 0])
    random_intercept_sd = np.sqrt(random_intercept_var)
    residual_sd = np.sqrt(result.scale)

    # ICC = var(random intercept) / (var(random intercept) + var(residual))
    icc = random_intercept_var / (random_intercept_var + result.scale)

    # Per-animal random effects
    animal_re = {str(k): float(v.iloc[0]) for k, v in result.random_effects.items()}

    # Fixed-effects covariance matrix (for CI computation)
    fe_cov = np.array(result.cov_params().iloc[:3, :3])

    # Marginal R² (proportion of variance explained by fixed effects)
    predicted_fixed = result.fe_params["Intercept"] + result.fe_params["log_weight"] * df["log_weight"] + result.fe_params["is_controlled"] * df["is_controlled"]
    ss_fixed = np.var(predicted_fixed)
    ss_total = np.var(df["log_dose"])
    r_squared_marginal = float(ss_fixed / ss_total) if ss_total > 0 else 0.0

    return MixedAllometricModel(
        a=a,
        b=b,
        a_se=a_se,
        b_se=b_se,
        control_effect=control_effect,
        control_effect_se=control_effect_se,
        a_controlled=a_controlled,
        n_animals=df["Animal_ID"].nunique(),
        random_intercept_sd=random_intercept_sd,
        residual_sd=residual_sd,
        icc=icc,
        animal_random_effects=animal_re,
        n_observations=len(df),
        log_likelihood=float(result.llf),
        aic=float(result.aic),
        bic=float(result.bic),
        r_squared_marginal=r_squared_marginal,
        log_a=log_a,
        log_a_se=log_a_se,
        _fe_cov=fe_cov,
    )
