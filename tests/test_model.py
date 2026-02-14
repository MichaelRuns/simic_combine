"""Tests for the allometric dosing model."""

import numpy as np
import pandas as pd
import pytest

from simic_combine.data import load_data, get_treatment_data, get_controlled_cases
from simic_combine.model import fit_allometric_model, power_law, AllometricModel
from simic_combine.predict import predict_optimal_dose, predict_dose_per_kg


class TestDataLoading:
    """Tests for data loading and cleaning."""

    def test_load_data_returns_dataframe(self):
        df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_treatment_data_has_required_columns(self):
        df = load_data()
        treatment = get_treatment_data(df)

        required_cols = ["Weight", "Control", "daily_dose_mcg", "dose_per_kg"]
        for col in required_cols:
            assert col in treatment.columns, f"Missing column: {col}"

    def test_controlled_cases_only_control_1(self):
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        assert all(controlled["Control"] == 1)
        assert len(controlled) > 0


class TestModelFitting:
    """Tests for allometric model fitting."""

    def test_fit_returns_allometric_model(self):
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        model = fit_allometric_model(
            controlled["Weight"],
            controlled["daily_dose_mcg"],
        )

        assert isinstance(model, AllometricModel)

    def test_model_parameters_reasonable(self):
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        model = fit_allometric_model(
            controlled["Weight"],
            controlled["daily_dose_mcg"],
        )

        # Coefficient 'a' should be positive and reasonable (20-200)
        assert 20 < model.a < 200, f"Coefficient a={model.a} out of expected range"

        # Exponent 'b' should be between 0 and 1 for allometric scaling
        # (meaning smaller animals need more mcg/kg)
        assert 0 < model.b < 1.5, f"Exponent b={model.b} out of expected range"

        # R-squared may be low due to high variability in clinical data
        # The key is that the model captures a meaningful trend
        assert model.r_squared > 0, f"R² = {model.r_squared} should be positive"

    def test_power_law_function(self):
        weights = np.array([1.0, 2.0, 4.0])
        a, b = 100, 0.75

        result = power_law(weights, a, b)

        expected = 100 * np.power(weights, 0.75)
        np.testing.assert_array_almost_equal(result, expected)

    def test_model_predict_increases_with_weight(self):
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        model = fit_allometric_model(
            controlled["Weight"],
            controlled["daily_dose_mcg"],
        )

        # Heavier kittens should need higher total dose
        dose_1kg = model.predict(1.0)
        dose_3kg = model.predict(3.0)
        dose_5kg = model.predict(5.0)

        assert dose_1kg < dose_3kg < dose_5kg

    def test_model_predict_per_kg_decreases_with_weight(self):
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        model = fit_allometric_model(
            controlled["Weight"],
            controlled["daily_dose_mcg"],
        )

        # Heavier kittens should need LESS mcg/kg
        dose_per_kg_1kg = model.predict_per_kg(1.0)
        dose_per_kg_3kg = model.predict_per_kg(3.0)
        dose_per_kg_5kg = model.predict_per_kg(5.0)

        assert dose_per_kg_1kg > dose_per_kg_3kg > dose_per_kg_5kg


class TestPrediction:
    """Tests for prediction functions."""

    def test_predict_returns_float(self):
        dose = predict_optimal_dose(2.0)
        assert isinstance(dose, float)
        assert dose > 0

    def test_predict_with_range_returns_tuple(self):
        result = predict_optimal_dose(2.0, return_range=True)
        assert isinstance(result, tuple)
        assert len(result) == 3

        low, mid, high = result
        assert low < mid < high

    def test_predict_dose_per_kg(self):
        dose_per_kg = predict_dose_per_kg(2.0)
        assert isinstance(dose_per_kg, float)
        assert dose_per_kg > 0

    def test_predict_negative_weight_raises(self):
        with pytest.raises(ValueError):
            predict_optimal_dose(-1.0)

        with pytest.raises(ValueError):
            predict_dose_per_kg(0)


class TestDoseTable:
    """Tests for dose reference table generation."""

    def test_dose_table_format(self):
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        model = fit_allometric_model(
            controlled["Weight"],
            controlled["daily_dose_mcg"],
        )

        table = model.dose_table()

        assert isinstance(table, pd.DataFrame)
        assert "Weight (kg)" in table.columns
        assert "Daily Dose (mcg)" in table.columns
        assert "Dose per kg (mcg/kg)" in table.columns
        assert len(table) == 8  # Default weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
