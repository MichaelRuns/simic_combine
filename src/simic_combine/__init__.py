"""Kitten hypothyroid dosing prediction using allometric scaling."""

from simic_combine.predict import predict_optimal_dose
from simic_combine.model import fit_allometric_model, AllometricModel

__version__ = "0.1.0"
__all__ = ["predict_optimal_dose", "fit_allometric_model", "AllometricModel"]
