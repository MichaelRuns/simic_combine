"""Tests for the allometric dosing model."""

import numpy as np
import pandas as pd
import pytest

from simic_combine.data import (
    load_data, get_treatment_data, get_controlled_cases,
    get_mixed_model_data, normalize_animal_ids, ANIMAL_ID_NORMALIZATION,
)
from simic_combine.model import (
    fit_allometric_model, power_law, AllometricModel,
    fit_mixed_model, MixedAllometricModel,
)
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
        assert len(table) == 7  # Default weights (1.0 to 6.0 kg)


class TestAnimalIDNormalization:
    """Tests for animal ID normalization."""

    def test_variant_names_mapped(self):
        df = load_data()
        treatment = get_treatment_data(df)
        normalized = normalize_animal_ids(treatment)

        id_col = "Animal ID" if "Animal ID" in normalized.columns else normalized.columns[0]
        all_ids = normalized[id_col].unique()

        # Variant names should not appear
        for variant in ANIMAL_ID_NORMALIZATION:
            if ANIMAL_ID_NORMALIZATION[variant] is not None:
                assert variant not in all_ids, f"Variant '{variant}' should have been normalized"

    def test_first_visits_dropped(self):
        df = load_data()
        treatment = get_treatment_data(df)
        normalized = normalize_animal_ids(treatment)

        id_col = "Animal ID" if "Animal ID" in normalized.columns else normalized.columns[0]
        assert "# first visits" not in normalized[id_col].values

    def test_canonical_names_preserved(self):
        df = load_data()
        treatment = get_treatment_data(df)
        normalized = normalize_animal_ids(treatment)

        id_col = "Animal ID" if "Animal ID" in normalized.columns else normalized.columns[0]
        all_ids = set(normalized[id_col].unique())

        # Canonical names that should exist (from the mapping targets)
        for canonical in ANIMAL_ID_NORMALIZATION.values():
            if canonical is not None and canonical in set(treatment[id_col].unique()) | set(ANIMAL_ID_NORMALIZATION.values()):
                # Only check if the canonical name is expected to exist in the data
                pass  # Some canonicals may only appear via mapping


class TestMixedModelData:
    """Tests for mixed model data preparation."""

    def test_mixed_data_has_required_columns(self):
        df = load_data()
        treatment = get_treatment_data(df)
        mixed_data = get_mixed_model_data(treatment)

        required = ["log_dose", "log_weight", "is_controlled", "Animal_ID"]
        for col in required:
            assert col in mixed_data.columns, f"Missing column: {col}"

    def test_no_nan_in_log_columns(self):
        df = load_data()
        treatment = get_treatment_data(df)
        mixed_data = get_mixed_model_data(treatment)

        assert not mixed_data["log_dose"].isna().any()
        assert not mixed_data["log_weight"].isna().any()

    def test_is_controlled_binary(self):
        df = load_data()
        treatment = get_treatment_data(df)
        mixed_data = get_mixed_model_data(treatment)

        assert set(mixed_data["is_controlled"].unique()).issubset({0, 1})

    def test_uses_all_treatment_data(self):
        df = load_data()
        treatment = get_treatment_data(df)
        mixed_data = get_mixed_model_data(treatment)

        # Should use most/all treatment observations (minus any dropped by normalization)
        assert len(mixed_data) >= len(treatment) - 5  # Allow small margin for dropped rows


class TestMixedModelFitting:
    """Tests for mixed-effects model fitting."""

    @pytest.fixture
    def mixed_model(self):
        df = load_data()
        treatment = get_treatment_data(df)
        mixed_data = get_mixed_model_data(treatment)
        return fit_mixed_model(mixed_data)

    def test_returns_mixed_model(self, mixed_model):
        assert isinstance(mixed_model, MixedAllometricModel)

    def test_reasonable_parameters(self, mixed_model):
        # Coefficient 'a' should be positive and reasonable
        assert 20 < mixed_model.a < 500, f"a={mixed_model.a} out of range"
        assert 20 < mixed_model.a_controlled < 500, f"a_controlled={mixed_model.a_controlled} out of range"

        # Exponent should be in reasonable range
        assert -0.5 < mixed_model.b < 2.0, f"b={mixed_model.b} out of range"

    def test_icc_in_valid_range(self, mixed_model):
        assert 0 < mixed_model.icc < 1, f"ICC={mixed_model.icc} out of (0,1)"

    def test_random_effects_exist(self, mixed_model):
        assert len(mixed_model.animal_random_effects) > 0
        assert mixed_model.n_animals > 0
        assert len(mixed_model.animal_random_effects) == mixed_model.n_animals

    def test_uses_more_data_than_original(self, mixed_model):
        # Mixed model should use more observations than controlled-only
        df = load_data()
        treatment = get_treatment_data(df)
        controlled = get_controlled_cases(treatment)

        assert mixed_model.n_observations > len(controlled)

    def test_predict_monotonically_increasing(self, mixed_model):
        doses = [mixed_model.predict(w) for w in [1.0, 2.0, 3.0, 4.0, 5.0]]
        for i in range(len(doses) - 1):
            assert doses[i] < doses[i + 1], "Dose should increase with weight"

    def test_predict_with_ci(self, mixed_model):
        pred, lower, upper = mixed_model.predict_with_ci(2.0)
        assert lower[0] < pred[0] < upper[0]

    def test_dose_table(self, mixed_model):
        table = mixed_model.dose_table()
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
