"""Kitten hypothyroid dosing prediction using allometric scaling."""

from simic_combine.predict import predict_optimal_dose, predict_optimal_dose_mixed
from simic_combine.model import fit_allometric_model, AllometricModel, fit_mixed_model, MixedAllometricModel

__version__ = "0.1.0"
__all__ = [
    "predict_optimal_dose",
    "predict_optimal_dose_mixed",
    "fit_allometric_model",
    "AllometricModel",
    "fit_mixed_model",
    "MixedAllometricModel",
]
